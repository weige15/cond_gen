from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


ANIMAL_LABELS = (
    "shark",
    "crocodile",
    "frog",
    "cat",
    "dog",
    "capybara",
    "elephant",
    "bird",
    "fish",
    "monkey",
)

OBJECT_LABELS = (
    "sneaker",
    "airplane",
    "coffee cup",
    "banana",
    "cactus",
    "toilet",
    "pizza",
    "drum",
    "car",
    "chair",
)

EXPECTED_TRAIN_ROWS = 4799
EXPECTED_GENERATE_ROWS = 2000
IMAGE_SIZE = 64


@dataclass(frozen=True)
class LabelMaps:
    animal_to_id: dict[str, int]
    object_to_id: dict[str, int]
    id_to_animal: tuple[str, ...]
    id_to_object: tuple[str, ...]


@dataclass(frozen=True)
class TrainRow:
    image_id: str
    animal: str
    object: str
    animal_id: int
    object_id: int


@dataclass(frozen=True)
class GenerateRow:
    image_id: str
    animal: str
    object: str
    prompt: str
    animal_id: int
    object_id: int


def build_label_maps() -> LabelMaps:
    return LabelMaps(
        animal_to_id={label: idx for idx, label in enumerate(ANIMAL_LABELS)},
        object_to_id={label: idx for idx, label in enumerate(OBJECT_LABELS)},
        id_to_animal=ANIMAL_LABELS,
        id_to_object=OBJECT_LABELS,
    )


def load_train_rows(csv_path: str | Path, label_maps: LabelMaps | None = None) -> list[TrainRow]:
    label_maps = label_maps or build_label_maps()
    rows: list[TrainRow] = []
    seen: set[str] = set()
    path = Path(csv_path)

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        _require_columns(path, reader, ("id", "animal", "object"))
        for raw in reader:
            line = reader.line_num
            image_id = _required_value(path, line, raw, "id")
            animal = _required_value(path, line, raw, "animal")
            object_name = _required_value(path, line, raw, "object")
            _reject_duplicate(path, line, image_id, seen)
            rows.append(
                TrainRow(
                    image_id=image_id,
                    animal=animal,
                    object=object_name,
                    animal_id=_label_id(path, line, "animal", animal, image_id, label_maps.animal_to_id),
                    object_id=_label_id(path, line, "object", object_name, image_id, label_maps.object_to_id),
                )
            )
    return rows


def load_generate_rows(csv_path: str | Path, label_maps: LabelMaps | None = None) -> list[GenerateRow]:
    label_maps = label_maps or build_label_maps()
    rows: list[GenerateRow] = []
    seen: set[str] = set()
    path = Path(csv_path)

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        _require_columns(path, reader, ("id", "animal", "object", "prompt"))
        for raw in reader:
            line = reader.line_num
            image_id = _required_value(path, line, raw, "id")
            animal = _required_value(path, line, raw, "animal")
            object_name = _required_value(path, line, raw, "object")
            prompt = _required_value(path, line, raw, "prompt")
            _reject_duplicate(path, line, image_id, seen)
            rows.append(
                GenerateRow(
                    image_id=image_id,
                    animal=animal,
                    object=object_name,
                    prompt=prompt,
                    animal_id=_label_id(path, line, "animal", animal, image_id, label_maps.animal_to_id),
                    object_id=_label_id(path, line, "object", object_name, image_id, label_maps.object_to_id),
                )
            )
    return rows


def validate_train_rows(
    rows: Iterable[TrainRow],
    image_dir: str | Path,
    *,
    expected_count: int | None = None,
    require_all_labels: bool = False,
    decode_images: bool = False,
    image_size: int = IMAGE_SIZE,
) -> None:
    row_list = list(rows)
    _validate_count("train rows", row_list, expected_count)
    _validate_label_coverage(row_list, require_all_labels)

    root = Path(image_dir)
    if not root.is_dir():
        raise ValueError(f"image directory not found: {root}")

    for row in row_list:
        image_path = root / row.image_id
        if not image_path.is_file():
            raise ValueError(f"missing training image for {row.image_id!r}: {image_path}")
        if decode_images:
            validate_image_file(image_path, image_size=image_size)


def validate_generate_rows(
    rows: Iterable[GenerateRow],
    *,
    expected_count: int | None = None,
    require_all_labels: bool = False,
) -> None:
    row_list = list(rows)
    _validate_count("generate rows", row_list, expected_count)
    _validate_label_coverage(row_list, require_all_labels)


def validate_image_file(image_path: str | Path, *, image_size: int = IMAGE_SIZE) -> None:
    path = Path(image_path)
    try:
        with Image.open(path) as image:
            if image.size != (image_size, image_size):
                raise ValueError(f"{path}: expected {image_size}x{image_size}, got {image.size}")
            image.convert("RGB")
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001 - keep CLI errors readable without a traceback.
        raise ValueError(f"{path}: cannot decode image: {exc}") from exc


def load_image_tensor(image_path: str | Path, *, image_size: int = IMAGE_SIZE):
    import numpy as np
    import torch

    path = Path(image_path)
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (image_size, image_size):
            raise ValueError(f"{path}: expected {image_size}x{image_size}, got {image.size}")
        array = np.array(image, dtype=np.float32, copy=True)
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor / 127.5 - 1.0


class BrainrotDataset:
    def __init__(self, rows: Iterable[TrainRow], image_dir: str | Path, *, image_size: int = IMAGE_SIZE):
        self.rows = list(rows)
        self.image_dir = Path(image_dir)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        import torch

        row = self.rows[index]
        return {
            "image": load_image_tensor(self.image_dir / row.image_id, image_size=self.image_size),
            "animal_id": torch.tensor(row.animal_id, dtype=torch.long),
            "object_id": torch.tensor(row.object_id, dtype=torch.long),
            "image_id": row.image_id,
        }


def check_assignment_data(
    train_csv: str | Path,
    generate_csv: str | Path,
    image_dir: str | Path,
    *,
    decode_images: bool = False,
) -> tuple[list[TrainRow], list[GenerateRow]]:
    label_maps = build_label_maps()
    train_rows = load_train_rows(train_csv, label_maps)
    generate_rows = load_generate_rows(generate_csv, label_maps)
    validate_train_rows(
        train_rows,
        image_dir,
        expected_count=EXPECTED_TRAIN_ROWS,
        require_all_labels=True,
        decode_images=decode_images,
    )
    validate_generate_rows(
        generate_rows,
        expected_count=EXPECTED_GENERATE_ROWS,
        require_all_labels=True,
    )
    return train_rows, generate_rows


def _require_columns(path: Path, reader: csv.DictReader, required: tuple[str, ...]) -> None:
    if reader.fieldnames is None:
        raise ValueError(f"{path}: missing CSV header")
    missing = [column for column in required if column not in reader.fieldnames]
    if missing:
        raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")


def _required_value(path: Path, line: int, row: dict[str, str | None], column: str) -> str:
    value = row.get(column)
    if value is None:
        raise ValueError(f"{path}:{line}: missing {column!r}")
    value = value.strip()
    if not value:
        raise ValueError(f"{path}:{line}: empty {column!r}")
    return value


def _reject_duplicate(path: Path, line: int, image_id: str, seen: set[str]) -> None:
    if image_id in seen:
        raise ValueError(f"{path}:{line}: duplicate id {image_id!r}")
    seen.add(image_id)


def _label_id(path: Path, line: int, kind: str, label: str, image_id: str, mapping: dict[str, int]) -> int:
    try:
        return mapping[label]
    except KeyError as exc:
        raise ValueError(f"{path}:{line}: unknown {kind} {label!r} for id {image_id!r}") from exc


def _validate_count(name: str, rows: list[object], expected_count: int | None) -> None:
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"{name}: expected {expected_count}, got {len(rows)}")


def _validate_label_coverage(rows: list[TrainRow] | list[GenerateRow], require_all_labels: bool) -> None:
    animal_labels = {row.animal for row in rows}
    object_labels = {row.object for row in rows}
    unknown_animals = animal_labels - set(ANIMAL_LABELS)
    unknown_objects = object_labels - set(OBJECT_LABELS)
    if unknown_animals:
        raise ValueError(f"unknown animal labels: {sorted(unknown_animals)}")
    if unknown_objects:
        raise ValueError(f"unknown object labels: {sorted(unknown_objects)}")
    if require_all_labels and animal_labels != set(ANIMAL_LABELS):
        raise ValueError(f"animal coverage mismatch: {sorted(animal_labels)}")
    if require_all_labels and object_labels != set(OBJECT_LABELS):
        raise ValueError(f"object coverage mismatch: {sorted(object_labels)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate HW6 Brainrot CSV labels and training image files.")
    parser.add_argument("--train_csv", type=Path)
    parser.add_argument("--generate_csv", type=Path)
    parser.add_argument("--image_dir", type=Path)
    parser.add_argument("--full", action="store_true", help="Require assignment row counts and all labels.")
    parser.add_argument("--decode_images", action="store_true", help="Open each training image and verify 64x64.")
    parser.add_argument("--expected_train_count", type=int)
    parser.add_argument("--expected_generate_count", type=int)
    args = parser.parse_args(argv)

    try:
        label_maps = build_label_maps()
        if args.full:
            if not args.train_csv or not args.generate_csv or not args.image_dir:
                parser.error("--full requires --train_csv, --generate_csv, and --image_dir")
            train_rows, generate_rows = check_assignment_data(
                args.train_csv,
                args.generate_csv,
                args.image_dir,
                decode_images=args.decode_images,
            )
        else:
            train_rows = load_train_rows(args.train_csv, label_maps) if args.train_csv else []
            generate_rows = load_generate_rows(args.generate_csv, label_maps) if args.generate_csv else []
            if args.train_csv:
                if not args.image_dir:
                    parser.error("--train_csv requires --image_dir for validation")
                validate_train_rows(
                    train_rows,
                    args.image_dir,
                    expected_count=args.expected_train_count,
                    decode_images=args.decode_images,
                )
            if args.generate_csv:
                validate_generate_rows(generate_rows, expected_count=args.expected_generate_count)
    except (OSError, ValueError) as exc:
        print(f"data check failed: {exc}", file=sys.stderr)
        return 1

    if args.train_csv:
        print(f"train rows: {len(train_rows)}")
    if args.generate_csv:
        print(f"generate rows: {len(generate_rows)}")
    print(f"animals: {len(ANIMAL_LABELS)}; objects: {len(OBJECT_LABELS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
