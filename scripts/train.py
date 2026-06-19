from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

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
    if args.steps < 1 and not args.data_check_only:
        raise ValueError("--steps must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad_accum_steps must be >= 1")
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
    print(f"train rows: {len(rows)}")
    if args.data_check_only:
        return 0

    if args.overfit_subset is not None:
        rows = rows[: args.overfit_subset]
        if not rows:
            raise ValueError("--overfit_subset selected zero rows")

    device = resolve_device(args.device)
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ema_state = clone_state_dict(model) if args.ema_decay > 0 else None
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    dataset = BrainrotDataset(rows, args.image_dir, image_size=model_config.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    step = 0
    last_loss = None
    micro_step = 0
    while step < args.steps:
        for batch in loader:
            images = batch["image"].to(device)
            animal_ids = batch["animal_id"].to(device)
            object_ids = batch["object_id"].to(device)
            if micro_step % args.grad_accum_steps == 0:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                loss = diffusion.training_loss(model, images, animal_ids, object_ids)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step + 1}: {loss.item()}")
            scaler.scale(loss / args.grad_accum_steps).backward()
            micro_step += 1
            last_loss = float(loss.detach().cpu())
            if micro_step % args.grad_accum_steps != 0:
                continue
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if ema_state is not None:
                update_ema(ema_state, model, args.ema_decay)
            step += 1
            if step >= args.steps:
                break

    payload = {
        "format_version": CHECKPOINT_VERSION,
        "model_state": model.state_dict(),
        "model_config": model_config_to_dict(model_config),
        "diffusion_config": diffusion_config_to_dict(diffusion_config),
        "label_maps": label_maps_to_dict(label_maps),
        "seed": args.seed,
        "training": {
            "train_csv": str(args.train_csv),
            "image_dir": str(args.image_dir),
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "grad_accum_steps": args.grad_accum_steps,
            "grad_clip": args.grad_clip,
            "amp": args.amp,
            "overfit_subset": args.overfit_subset,
            "last_loss": last_loss,
            "ema_decay": args.ema_decay,
        },
    }
    if ema_state is not None:
        payload["ema_state"] = {key: value.cpu() for key, value in ema_state.items()}
    save_checkpoint(payload, args.checkpoint)
    print(f"saved checkpoint: {args.checkpoint}")
    print(f"last loss: {last_loss:.6f}")
    return 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
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


@torch.no_grad()
def update_ema(ema_state: dict[str, torch.Tensor], model: nn.Module, decay: float) -> None:
    current = model.state_dict()
    for key, value in current.items():
        ema_state[key].mul_(decay).add_(value.detach(), alpha=1.0 - decay)


if __name__ == "__main__":
    raise SystemExit(main())
