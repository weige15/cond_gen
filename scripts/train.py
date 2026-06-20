from __future__ import annotations

import argparse
import contextlib
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from scripts.checkpoint import CHECKPOINT_VERSION, label_maps_to_dict, save_checkpoint
from scripts.data import (
    EXPECTED_TRAIN_ROWS,
    BrainrotDataset,
    build_label_maps,
    load_train_rows,
    validate_train_rows,
)
from scripts.model import (
    DiffusionConfig,
    ModelConfig,
    build_diffusion,
    build_model,
    diffusion_config_to_dict,
    model_config_to_dict,
)


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


@dataclass
class EarlyStopState:
    best_loss: float | None = None
    stale_checks: int = 0
    logged_losses: list[float] = field(default_factory=list)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the HW6 conditional DDPM from scratch.")
    parser.add_argument("--train_csv", type=Path, required=True)
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("model.pth"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--checkpoint_every", type=int, default=10000)
    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    parser.add_argument("--early_stop_window", type=int, default=5)
    parser.add_argument("--early_stop_warmup_steps", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--overfit_subset", type=int)
    parser.add_argument("--data_check_only", action="store_true")
    parser.add_argument("--full_data_check", action="store_true")
    parser.add_argument("--decode_images", action="store_true")
    parser.add_argument("--expected_train_count", type=int)
    parser.add_argument("--base_channels", type=int, default=128)
    parser.add_argument("--channel_mults", default="1,2,3,4")
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--attention_resolutions", default="16,8")
    parser.add_argument("--attention_heads", type=int, default=4)
    parser.add_argument("--cond_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--condition_dropout", type=float, default=0.1)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--schedule", choices=("cosine", "linear"), default="cosine")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"training failed: {exc}", file=sys.stderr)
        return 1


def run(args: argparse.Namespace) -> int:
    distributed = init_distributed()
    if args.steps < 1 and not args.data_check_only:
        raise ValueError("--steps must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad_accum_steps must be >= 1")
    if args.log_every < 1:
        raise ValueError("--log_every must be >= 1")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint_every must be >= 0")
    if args.early_stop_patience < 0:
        raise ValueError("--early_stop_patience must be >= 0")
    if args.early_stop_min_delta < 0:
        raise ValueError("--early_stop_min_delta must be >= 0")
    if args.early_stop_window < 1:
        raise ValueError("--early_stop_window must be >= 1")
    if args.early_stop_warmup_steps < 0:
        raise ValueError("--early_stop_warmup_steps must be >= 0")
    if not 0 <= args.condition_dropout < 1:
        raise ValueError("--condition_dropout must be in [0, 1)")

    set_seed(args.seed)
    label_maps = build_label_maps()
    rows = load_train_rows(args.train_csv, label_maps)
    expected_count = args.expected_train_count
    if args.full_data_check and expected_count is None:
        expected_count = EXPECTED_TRAIN_ROWS
    validate_train_rows(
        rows,
        args.image_dir,
        expected_count=expected_count,
        require_all_labels=args.full_data_check,
        decode_images=args.decode_images,
    )
    log(distributed, f"train rows: {len(rows)}")
    if args.data_check_only:
        return 0

    if args.overfit_subset is not None:
        rows = rows[: args.overfit_subset]
        if not rows:
            raise ValueError("--overfit_subset selected zero rows")

    device = resolve_device(args.device, distributed)
    model_config = ModelConfig(
        base_channels=args.base_channels,
        channel_mults=parse_int_tuple(args.channel_mults),
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=parse_int_tuple(args.attention_resolutions),
        attention_heads=args.attention_heads,
        cond_dim=args.cond_dim,
        dropout=args.dropout,
        condition_dropout=args.condition_dropout,
    )
    diffusion_config = DiffusionConfig(timesteps=args.timesteps, schedule=args.schedule)
    model = build_model(model_config).to(device)
    diffusion = build_diffusion(diffusion_config)
    if distributed.enabled:
        ddp_kwargs = {"find_unused_parameters": args.condition_dropout == 0}
        if device.type == "cuda":
            ddp_kwargs["device_ids"] = [distributed.local_rank]
        model = DistributedDataParallel(model, **ddp_kwargs)
        set_seed(args.seed + distributed.rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ema_state = clone_state_dict(unwrap_model(model)) if args.ema_decay > 0 else None
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    global_batch_size = args.batch_size * distributed.world_size * args.grad_accum_steps

    dataset = BrainrotDataset(rows, args.image_dir, image_size=model_config.image_size)
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=distributed.world_size,
            rank=distributed.rank,
            shuffle=True,
            seed=args.seed,
        )
        if distributed.enabled
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=False,
    )
    step = 0
    last_loss = None
    micro_step = 0
    epoch = 0
    start_time = time.monotonic()
    early_stop_state = EarlyStopState()
    stop_reason = None
    log(
        distributed,
        f"training: steps={args.steps} per_gpu_batch={args.batch_size} "
        f"global_batch={global_batch_size} world_size={distributed.world_size}",
    )
    while step < args.steps and stop_reason is None:
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch += 1
        for batch in loader:
            images = batch["image"].to(device)
            animal_ids = batch["animal_id"].to(device)
            object_ids = batch["object_id"].to(device)
            if micro_step % args.grad_accum_steps == 0:
                optimizer.zero_grad(set_to_none=True)
            should_step = (micro_step + 1) % args.grad_accum_steps == 0
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                loss = diffusion.training_loss(model, images, animal_ids, object_ids)
            if not is_finite_loss(loss, distributed):
                raise RuntimeError(f"non-finite loss at step {step + 1}: {loss.item()}")
            sync_context = (
                model.no_sync()
                if isinstance(model, DistributedDataParallel) and not should_step
                else contextlib.nullcontext()
            )
            with sync_context:
                scaler.scale(loss / args.grad_accum_steps).backward()
            micro_step += 1
            last_loss = float(loss.detach().cpu())
            if not should_step:
                continue
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if ema_state is not None:
                update_ema(ema_state, unwrap_model(model), args.ema_decay)
            step += 1
            if step == 1 or step % args.log_every == 0 or step >= args.steps:
                loss_value = mean_loss(loss.detach(), distributed)
                log_progress(distributed, step, args.steps, loss_value, start_time)
                stop_reason = maybe_stop_early(early_stop_state, args, step, loss_value, distributed, device)
            if args.checkpoint_every and step % args.checkpoint_every == 0 and distributed.is_main:
                checkpoint_path = checkpoint_path_for_step(args.checkpoint, step)
                save_training_checkpoint(
                    args=args,
                    path=checkpoint_path,
                    distributed=distributed,
                    label_maps=label_maps,
                    model=model,
                    model_config=model_config,
                    diffusion_config=diffusion_config,
                    ema_state=ema_state,
                    global_batch_size=global_batch_size,
                    completed_steps=step,
                    last_loss=last_loss,
                    stop_reason=None,
                )
                print(f"saved checkpoint: {checkpoint_path}")
            if step >= args.steps or stop_reason is not None:
                break

    if distributed.is_main:
        save_training_checkpoint(
            args=args,
            path=args.checkpoint,
            distributed=distributed,
            label_maps=label_maps,
            model=model,
            model_config=model_config,
            diffusion_config=diffusion_config,
            ema_state=ema_state,
            global_batch_size=global_batch_size,
            completed_steps=step,
            last_loss=last_loss,
            stop_reason=stop_reason,
        )
        print(f"saved checkpoint: {args.checkpoint}")
        if stop_reason is not None:
            print(stop_reason)
        print(f"last loss: {last_loss:.6f}")
    if distributed.enabled:
        dist.barrier()
    return 0


def init_distributed() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistributedContext(enabled=False)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend)
    return DistributedContext(enabled=True, rank=rank, local_rank=local_rank, world_size=world_size)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str, distributed: DistributedContext | None = None) -> torch.device:
    distributed = distributed or DistributedContext(enabled=False)
    if distributed.enabled:
        if torch.cuda.is_available():
            if name not in ("auto", "cuda"):
                raise ValueError("distributed CUDA training expects --device auto or --device cuda")
            return torch.device(f"cuda:{distributed.local_rank}")
        if name not in ("auto", "cpu"):
            raise ValueError("distributed CPU training expects --device auto or --device cpu")
        return torch.device("cpu")

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available")
    return device


def parse_int_tuple(value: str) -> tuple[int, ...]:
    value = value.strip()
    if not value:
        return ()
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def log(distributed: DistributedContext, message: str) -> None:
    if distributed.is_main:
        print(message)


def log_progress(
    distributed: DistributedContext,
    step: int,
    total_steps: int,
    loss: float,
    start_time: float,
) -> None:
    if not distributed.is_main:
        return
    elapsed = time.monotonic() - start_time
    steps_per_second = step / elapsed if elapsed > 0 else 0.0
    remaining_steps = max(total_steps - step, 0)
    eta = remaining_steps / steps_per_second if steps_per_second > 0 else 0.0
    percent = 100.0 * step / total_steps
    print(
        f"step {step}/{total_steps} ({percent:.1f}%) "
        f"loss={loss:.6f} elapsed={format_duration(elapsed)} "
        f"eta={format_duration(eta)} speed={steps_per_second:.3f} steps/s",
        flush=True,
    )


def maybe_stop_early(
    state: EarlyStopState,
    args: argparse.Namespace,
    step: int,
    loss: float,
    distributed: DistributedContext,
    device: torch.device,
) -> str | None:
    reason = None
    if distributed.is_main and args.early_stop_patience > 0:
        state.logged_losses.append(loss)
        if step >= args.early_stop_warmup_steps and len(state.logged_losses) >= args.early_stop_window:
            window = state.logged_losses[-args.early_stop_window :]
            window_loss = sum(window) / len(window)
            if state.best_loss is None or window_loss < state.best_loss - args.early_stop_min_delta:
                state.best_loss = window_loss
                state.stale_checks = 0
            else:
                state.stale_checks += 1
                if state.stale_checks >= args.early_stop_patience:
                    reason = (
                        f"early stopping at step {step}: loss window average {window_loss:.6f} "
                        f"did not improve by {args.early_stop_min_delta:g} for "
                        f"{args.early_stop_patience} logged checks"
                    )
    if not distributed.enabled:
        return reason
    should_stop = torch.tensor(1 if reason is not None else 0, device=device)
    dist.broadcast(should_stop, src=0)
    if should_stop.item():
        return reason or "early stopping requested by rank 0"
    return None


def save_training_checkpoint(
    *,
    args: argparse.Namespace,
    path: Path,
    distributed: DistributedContext,
    label_maps,
    model: nn.Module,
    model_config: ModelConfig,
    diffusion_config: DiffusionConfig,
    ema_state: dict[str, torch.Tensor] | None,
    global_batch_size: int,
    completed_steps: int,
    last_loss: float | None,
    stop_reason: str | None,
) -> None:
    model_for_save = unwrap_model(model)
    payload = {
        "format_version": CHECKPOINT_VERSION,
        "model_state": model_for_save.state_dict(),
        "model_config": model_config_to_dict(model_config),
        "diffusion_config": diffusion_config_to_dict(diffusion_config),
        "label_maps": label_maps_to_dict(label_maps),
        "seed": args.seed,
        "training": {
            "train_csv": str(args.train_csv),
            "image_dir": str(args.image_dir),
            "steps": completed_steps,
            "target_steps": args.steps,
            "batch_size": args.batch_size,
            "global_batch_size": global_batch_size,
            "lr": args.lr,
            "grad_accum_steps": args.grad_accum_steps,
            "grad_clip": args.grad_clip,
            "amp": args.amp,
            "overfit_subset": args.overfit_subset,
            "last_loss": last_loss,
            "ema_decay": args.ema_decay,
            "distributed": distributed.enabled,
            "world_size": distributed.world_size,
            "checkpoint_every": args.checkpoint_every,
            "early_stop_patience": args.early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
            "early_stop_window": args.early_stop_window,
            "early_stop_warmup_steps": args.early_stop_warmup_steps,
            "early_stop_reason": stop_reason,
        },
    }
    if ema_state is not None:
        payload["ema_state"] = {key: value.cpu() for key, value in ema_state.items()}
    save_checkpoint(payload, path)


def checkpoint_path_for_step(path: Path, step: int) -> Path:
    return path.with_name(f"{path.stem}_step{step:06d}{path.suffix}")


def format_duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
    return f"{minutes:d}m{seconds:02d}s"


def is_finite_loss(loss: torch.Tensor, distributed: DistributedContext) -> bool:
    finite = torch.isfinite(loss).to(dtype=torch.int, device=loss.device)
    if distributed.enabled:
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    return bool(finite.item())


def mean_loss(loss: torch.Tensor, distributed: DistributedContext) -> float:
    value = loss.float()
    if distributed.enabled:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= distributed.world_size
    return float(value.cpu())


@torch.no_grad()
def update_ema(ema_state: dict[str, torch.Tensor], model: nn.Module, decay: float) -> None:
    current = model.state_dict()
    for key, value in current.items():
        ema_state[key].mul_(decay).add_(value.detach(), alpha=1.0 - decay)


if __name__ == "__main__":
    raise SystemExit(main())
