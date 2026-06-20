from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from PIL import Image
import torch
import torch.multiprocessing as mp
from torch.multiprocessing.spawn import ProcessExitedException, ProcessRaisedException

from scripts.checkpoint import load_checkpoint
from scripts.data import (
    EXPECTED_GENERATE_ROWS,
    IMAGE_SIZE,
    GenerateRow,
    build_label_maps,
    load_generate_rows,
    validate_generate_rows,
)
from scripts.model import (
    build_diffusion,
    build_model,
    diffusion_config_from_dict,
    images_to_uint8,
    model_config_from_dict,
)
from scripts.train import resolve_device, set_seed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate HW6 Brainrot images from a trained checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--generate_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("generated_images"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--devices",
        help="Comma-separated CUDA devices for sharded generation, e.g. auto, 0,1, or cuda:0,cuda:1.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--expected_count", type=int, default=EXPECTED_GENERATE_ROWS)
    parser.add_argument("--sampling_steps", type=int)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except (OSError, ValueError, RuntimeError, ProcessExitedException, ProcessRaisedException) as exc:
        print(f"generation failed: {exc}", file=sys.stderr)
        return 1


def run(args: argparse.Namespace) -> int:
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    devices = _parse_devices(args.devices)
    if devices is not None:
        return _run_multi_device(args, devices)

    set_seed(args.seed)
    device = resolve_device(args.device)
    _activate_device(device)
    payload, model_config, diffusion_config, label_maps = _load_checkpoint_parts(args, map_location=device)
    rows = _load_rows(args, label_maps)
    _prepare_output_dir(args.output_dir, {row.image_id for row in rows}, overwrite=args.overwrite)
    written = _write_rows(args, rows, payload, model_config, diffusion_config, device)

    print(f"generated {written} images in {args.output_dir}")
    return 0


def _run_multi_device(args: argparse.Namespace, devices: list[str]) -> int:
    if args.device != "auto":
        raise ValueError("--devices cannot be combined with --device")
    _check_cuda_devices(devices)
    if len(devices) == 1:
        set_seed(args.seed)
        device = torch.device(devices[0])
        _activate_device(device)
        payload, model_config, diffusion_config, label_maps = _load_checkpoint_parts(args, map_location=device)
        rows = _load_rows(args, label_maps)
        _prepare_output_dir(args.output_dir, {row.image_id for row in rows}, overwrite=args.overwrite)
        written = _write_rows(args, rows, payload, model_config, diffusion_config, device)
        print(f"generated {written} images in {args.output_dir}")
        return 0

    payload, _, _, label_maps = _load_checkpoint_parts(args, map_location="cpu")
    rows = _load_rows(args, label_maps)
    _prepare_output_dir(args.output_dir, {row.image_id for row in rows}, overwrite=args.overwrite)
    del payload

    shards = _shard_rows(rows, len(devices))
    mp.spawn(_multi_device_worker, args=(args, devices, shards), nprocs=len(devices), join=True)
    print(f"generated {len(rows)} images in {args.output_dir} using {len(devices)} devices")
    return 0


def _multi_device_worker(rank: int, args: argparse.Namespace, devices: list[str], shards: list[list[GenerateRow]]) -> None:
    device = torch.device(devices[rank])
    _activate_device(device)
    set_seed(args.seed + rank)
    payload, model_config, diffusion_config, _ = _load_checkpoint_parts(args, map_location=device)
    _write_rows(args, shards[rank], payload, model_config, diffusion_config, device)


def _load_checkpoint_parts(
    args: argparse.Namespace,
    *,
    map_location: str | torch.device,
):
    payload = load_checkpoint(args.checkpoint, map_location=map_location)
    model_config = model_config_from_dict(payload["model_config"])
    diffusion_config = diffusion_config_from_dict(payload["diffusion_config"])
    label_maps = build_label_maps()
    _check_label_maps(payload["label_maps"], label_maps)
    return payload, model_config, diffusion_config, label_maps


def _load_rows(args: argparse.Namespace, label_maps) -> list[GenerateRow]:
    rows = load_generate_rows(args.generate_csv, label_maps)
    validate_generate_rows(rows, expected_count=args.expected_count)
    return rows


def _write_rows(
    args: argparse.Namespace,
    rows: list[GenerateRow],
    payload: dict[str, object],
    model_config,
    diffusion_config,
    device: torch.device,
) -> int:
    model = build_model(model_config).to(device)
    state_key = "ema_state" if args.use_ema else "model_state"
    if state_key not in payload:
        raise ValueError(f"checkpoint missing {state_key}")
    model.load_state_dict(payload[state_key])
    model.eval()
    diffusion = build_diffusion(diffusion_config)

    written = 0
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        animal_ids = torch.tensor([row.animal_id for row in batch], dtype=torch.long, device=device)
        object_ids = torch.tensor([row.object_id for row in batch], dtype=torch.long, device=device)
        samples = diffusion.sample(
            model,
            animal_ids,
            object_ids,
            image_size=model_config.image_size,
            sampling_steps=args.sampling_steps,
            guidance_scale=args.guidance_scale,
            eta=args.ddim_eta,
        )
        arrays = images_to_uint8(samples).permute(0, 2, 3, 1).numpy()
        for row, array in zip(batch, arrays, strict=True):
            target = _target_path(args.output_dir, row.image_id)
            Image.fromarray(array).save(target)
            written += 1

    return written


def _parse_devices(value: str | None) -> list[str] | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        raise ValueError("--devices must not be empty")
    if value == "auto":
        if not torch.cuda.is_available():
            raise ValueError("--devices auto requires CUDA")
        return [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]

    devices = []
    for part in value.split(","):
        name = part.strip()
        if not name:
            continue
        if name.isdecimal():
            name = f"cuda:{name}"
        device = torch.device(name)
        if device.type != "cuda" or device.index is None:
            raise ValueError("--devices expects CUDA device ids like 0,1 or cuda:0,cuda:1")
        devices.append(str(device))
    if not devices:
        raise ValueError("--devices must include at least one device")
    if len(set(devices)) != len(devices):
        raise ValueError("--devices contains duplicate devices")
    return devices


def _check_cuda_devices(devices: list[str]) -> None:
    if not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available")
    count = torch.cuda.device_count()
    for name in devices:
        device = torch.device(name)
        if device.index is None or device.index >= count:
            raise ValueError(f"CUDA device not available: {name}")


def _activate_device(device: torch.device) -> None:
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)


def _shard_rows(rows: list[GenerateRow], parts: int) -> list[list[GenerateRow]]:
    if parts < 1:
        raise ValueError("parts must be >= 1")
    return [rows[rank::parts] for rank in range(parts)]


def _check_label_maps(checkpoint_maps: dict[str, object], label_maps) -> None:
    if checkpoint_maps["animal_to_id"] != label_maps.animal_to_id:
        raise ValueError("checkpoint animal label mapping is incompatible")
    if checkpoint_maps["object_to_id"] != label_maps.object_to_id:
        raise ValueError("checkpoint object label mapping is incompatible")


def _prepare_output_dir(output_dir: Path, expected_names: set[str], *, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = {path.name for path in output_dir.iterdir() if path.is_file()}
    if not existing:
        return
    extra = sorted(existing - expected_names)
    if extra:
        raise ValueError(f"output_dir has extra files: {', '.join(extra[:10])}")
    if not overwrite:
        raise ValueError("output_dir is not empty; pass --overwrite to replace requested files")


def _target_path(output_dir: Path, image_id: str) -> Path:
    path = Path(image_id)
    if path.name != image_id:
        raise ValueError(f"nested output paths are not supported: {image_id!r}")
    return output_dir / image_id


if __name__ == "__main__":
    raise SystemExit(main())
