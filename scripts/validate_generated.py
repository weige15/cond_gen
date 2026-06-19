from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from PIL import Image

from scripts.data import EXPECTED_GENERATE_ROWS, IMAGE_SIZE, load_generate_rows, validate_generate_rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated HW6 PNG outputs.")
    parser.add_argument("--generate_csv", type=Path, required=True)
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--expected_count", type=int, default=EXPECTED_GENERATE_ROWS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        errors = validate_generated(args.generate_csv, args.image_dir, args.image_size, args.expected_count)
    except (OSError, ValueError) as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        return 1

    if errors:
        print(f"validation failed with {len(errors)} error(s):", file=sys.stderr)
        for error in errors[:20]:
            print(f"- {error}", file=sys.stderr)
        if len(errors) > 20:
            print(f"- ... {len(errors) - 20} more", file=sys.stderr)
        return 1

    print(f"validated {args.expected_count} generated PNG files in {args.image_dir}")
    return 0


def validate_generated(
    generate_csv: str | Path,
    image_dir: str | Path,
    image_size: int = IMAGE_SIZE,
    expected_count: int = EXPECTED_GENERATE_ROWS,
) -> list[str]:
    rows = load_generate_rows(generate_csv)
    validate_generate_rows(rows, expected_count=expected_count)

    output_dir = Path(image_dir)
    if not output_dir.is_dir():
        raise ValueError(f"image directory not found: {output_dir}")

    expected = {row.image_id for row in rows}
    actual = {path.name for path in output_dir.iterdir() if path.is_file()}
    errors: list[str] = []

    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        errors.append(f"missing files: {', '.join(missing[:10])}")
    if extra:
        errors.append(f"extra files: {', '.join(extra[:10])}")

    for image_id in sorted(expected & actual):
        path = output_dir / image_id
        try:
            with Image.open(path) as image:
                image.load()
                if image.format != "PNG":
                    errors.append(f"{image_id}: expected PNG, got {image.format}")
                if image.mode != "RGB":
                    errors.append(f"{image_id}: expected RGB, got {image.mode}")
                if image.size != (image_size, image_size):
                    errors.append(f"{image_id}: expected {image_size}x{image_size}, got {image.size}")
        except Exception as exc:  # noqa: BLE001 - error text is the validator output.
            errors.append(f"{image_id}: cannot open image: {exc}")

    return errors


if __name__ == "__main__":
    raise SystemExit(main())
