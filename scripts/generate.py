from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from PIL import Image
import torch

from scripts.checkpoint import load_checkpoint
from scripts.data import EXPECTED_GENERATE_ROWS, IMAGE_SIZE, build_label_maps, load_generate_rows, validate_generate_rows
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
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"generation failed: {exc}", file=sys.stderr)
        return 1


def run(args: argparse.Namespace) -> int:
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    set_seed(args.seed)
    device = resolve_device(args.device)

    payload = load_checkpoint(args.checkpoint, map_location=device)
    model_config = model_config_from_dict(payload["model_config"])
    diffusion_config = diffusion_config_from_dict(payload["diffusion_config"])

    label_maps = build_label_maps()
    _check_label_maps(payload["label_maps"], label_maps)
    rows = load_generate_rows(args.generate_csv, label_maps)
    validate_generate_rows(rows, expected_count=args.expected_count)
    _prepare_output_dir(args.output_dir, {row.image_id for row in rows}, overwrite=args.overwrite)

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

    print(f"generated {written} images in {args.output_dir}")
    return 0


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
