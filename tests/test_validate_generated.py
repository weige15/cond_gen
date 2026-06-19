from __future__ import annotations

import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.validate_generated import main


class GeneratedValidatorTests(unittest.TestCase):
    def test_valid_fixture_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path, image_dir = self._fixture(Path(tmp), ["000001.png", "000002.png"])
            self._png(image_dir / "000001.png")
            self._png(image_dir / "000002.png")
            self.assertEqual(self._run(csv_path, image_dir, 2), 0)

    def test_missing_file_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path, image_dir = self._fixture(Path(tmp), ["000001.png"])
            self.assertNotEqual(self._run(csv_path, image_dir, 1), 0)

    def test_extra_file_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path, image_dir = self._fixture(Path(tmp), ["000001.png"])
            self._png(image_dir / "000001.png")
            self._png(image_dir / "extra.png")
            self.assertNotEqual(self._run(csv_path, image_dir, 1), 0)

    def test_duplicate_generation_id_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            image_dir.mkdir()
            csv_path = root / "generate.csv"
            self._write_generate_csv(csv_path, ["000001.png", "000001.png"])
            self._png(image_dir / "000001.png")
            self.assertNotEqual(self._run(csv_path, image_dir, 2), 0)

    def test_grayscale_and_rgba_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path, image_dir = self._fixture(Path(tmp), ["gray.png", "rgba.png"])
            self._png(image_dir / "gray.png", mode="L")
            self._png(image_dir / "rgba.png", mode="RGBA")
            self.assertNotEqual(self._run(csv_path, image_dir, 2), 0)

    def test_wrong_size_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path, image_dir = self._fixture(Path(tmp), ["000001.png"])
            self._png(image_dir / "000001.png", size=(32, 32))
            self.assertNotEqual(self._run(csv_path, image_dir, 1), 0)

    def test_non_png_data_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path, image_dir = self._fixture(Path(tmp), ["000001.png"])
            (image_dir / "000001.png").write_bytes(b"not a png")
            self.assertNotEqual(self._run(csv_path, image_dir, 1), 0)

    def _fixture(self, root: Path, names: list[str]) -> tuple[Path, Path]:
        image_dir = root / "images"
        image_dir.mkdir()
        csv_path = root / "generate.csv"
        self._write_generate_csv(csv_path, names)
        return csv_path, image_dir

    def _write_generate_csv(self, path: Path, names: list[str]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "animal", "object", "prompt"])
            for name in names:
                writer.writerow([name, "fish", "chair", "a fish and a chair"])

    def _png(self, path: Path, size: tuple[int, int] = (64, 64), mode: str = "RGB") -> None:
        Image.new(mode, size).save(path)

    def _run(self, csv_path: Path, image_dir: Path, expected_count: int) -> int:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return main(
                [
                    "--generate_csv",
                    str(csv_path),
                    "--image_dir",
                    str(image_dir),
                    "--expected_count",
                    str(expected_count),
                ]
            )


if __name__ == "__main__":
    unittest.main()
