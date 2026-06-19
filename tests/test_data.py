from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.data import (
    BrainrotDataset,
    build_label_maps,
    load_generate_rows,
    load_train_rows,
    validate_generate_rows,
    validate_train_rows,
)


class DataModuleTests(unittest.TestCase):
    def test_valid_rows_and_dataset_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            image_dir.mkdir()
            self._png(image_dir / "000001.png")
            train_csv = root / "train.csv"
            gen_csv = root / "generate.csv"
            self._write_csv(train_csv, ["id", "animal", "object"], [["000001.png", "fish", "chair"]])
            self._write_csv(
                gen_csv,
                ["id", "animal", "object", "prompt"],
                [["000002.png", "cat", "banana", "a cat and a banana"]],
            )

            train_rows = load_train_rows(train_csv)
            generate_rows = load_generate_rows(gen_csv)
            validate_train_rows(train_rows, image_dir, expected_count=1, decode_images=True)
            validate_generate_rows(generate_rows, expected_count=1)

            item = BrainrotDataset(train_rows, image_dir)[0]
            self.assertEqual(tuple(item["image"].shape), (3, 64, 64))
            self.assertEqual(int(item["animal_id"]), build_label_maps().animal_to_id["fish"])
            self.assertEqual(int(item["object_id"]), build_label_maps().object_to_id["chair"])

    def test_label_maps_are_stable(self) -> None:
        first = build_label_maps()
        second = build_label_maps()
        self.assertEqual(first.animal_to_id, second.animal_to_id)
        self.assertEqual(first.object_to_id, second.object_to_id)
        self.assertEqual(first.animal_to_id["shark"], 0)
        self.assertEqual(first.object_to_id["chair"], 9)

    def test_missing_required_column_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.csv"
            self._write_csv(path, ["id", "animal"], [["000001.png", "fish"]])
            with self.assertRaisesRegex(ValueError, "missing required columns: object"):
                load_train_rows(path)

    def test_duplicate_id_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.csv"
            rows = [["000001.png", "fish", "chair"], ["000001.png", "cat", "chair"]]
            self._write_csv(path, ["id", "animal", "object"], rows)
            with self.assertRaisesRegex(ValueError, "duplicate id '000001.png'"):
                load_train_rows(path)

    def test_unknown_label_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.csv"
            self._write_csv(path, ["id", "animal", "object"], [["000001.png", "dragon", "chair"]])
            with self.assertRaisesRegex(ValueError, "unknown animal 'dragon'"):
                load_train_rows(path)

    def test_empty_field_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "generate.csv"
            self._write_csv(path, ["id", "animal", "object", "prompt"], [["000001.png", "fish", "", "x"]])
            with self.assertRaisesRegex(ValueError, "empty 'object'"):
                load_generate_rows(path)

    def test_missing_image_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            image_dir.mkdir()
            train_csv = root / "train.csv"
            self._write_csv(train_csv, ["id", "animal", "object"], [["missing.png", "fish", "chair"]])
            rows = load_train_rows(train_csv)
            with self.assertRaisesRegex(ValueError, "missing training image"):
                validate_train_rows(rows, image_dir)

    def test_wrong_size_image_fails_when_decoded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            image_dir.mkdir()
            self._png(image_dir / "000001.png", size=(32, 32))
            train_csv = root / "train.csv"
            self._write_csv(train_csv, ["id", "animal", "object"], [["000001.png", "fish", "chair"]])
            rows = load_train_rows(train_csv)
            with self.assertRaisesRegex(ValueError, "expected 64x64"):
                validate_train_rows(rows, image_dir, decode_images=True)

    def _write_csv(self, path: Path, headers: list[str], rows: list[list[str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)

    def _png(self, path: Path, size: tuple[int, int] = (64, 64), mode: str = "RGB") -> None:
        Image.new(mode, size, (12, 34, 56)).save(path)


if __name__ == "__main__":
    unittest.main()
