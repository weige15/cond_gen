from __future__ import annotations

import csv
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.checkpoint import load_checkpoint
from scripts.train import main


class TrainingCliTests(unittest.TestCase):
    def test_data_check_only_passes_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, image_dir, train_csv = self._fixture(Path(tmp), rows=1)
            code = self._run(
                [
                    "--train_csv",
                    str(train_csv),
                    "--image_dir",
                    str(image_dir),
                    "--data_check_only",
                    "--expected_train_count",
                    "1",
                    "--decode_images",
                ]
            )
            self.assertEqual(code, 0)
            self.assertFalse((root / "model.pth").exists())

    def test_one_step_training_saves_required_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, image_dir, train_csv = self._fixture(Path(tmp), rows=2)
            checkpoint = root / "tiny.pth"
            code = self._run(
                [
                    "--train_csv",
                    str(train_csv),
                    "--image_dir",
                    str(image_dir),
                    "--checkpoint",
                    str(checkpoint),
                    "--steps",
                    "1",
                    "--batch_size",
                    "1",
                    "--device",
                    "cpu",
                    "--expected_train_count",
                    "2",
                    "--base_channels",
                    "4",
                    "--cond_dim",
                    "32",
                    "--channel_mults",
                    "1,2",
                    "--attention_resolutions",
                    "",
                    "--num_res_blocks",
                    "1",
                    "--timesteps",
                    "2",
                    "--schedule",
                    "linear",
                ]
            )
            self.assertEqual(code, 0)
            payload = load_checkpoint(checkpoint)
            self.assertIn("model_state", payload)
            self.assertEqual(payload["model_config"]["base_channels"], 4)
            self.assertEqual(payload["model_config"]["channel_mults"], (1, 2))
            self.assertEqual(payload["diffusion_config"]["timesteps"], 2)
            self.assertIn("ema_state", payload)
            self.assertEqual(payload["training"]["steps"], 1)

    def test_missing_image_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            image_dir.mkdir()
            train_csv = root / "train.csv"
            with train_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["id", "animal", "object"])
                writer.writerow(["missing.png", "fish", "chair"])
            code = self._run(["--train_csv", str(train_csv), "--image_dir", str(image_dir), "--data_check_only"])
            self.assertNotEqual(code, 0)

    def _fixture(self, root: Path, *, rows: int) -> tuple[Path, Path, Path]:
        image_dir = root / "images"
        image_dir.mkdir()
        train_csv = root / "train.csv"
        with train_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "animal", "object"])
            for idx in range(rows):
                name = f"{idx + 1:06d}.png"
                Image.new("RGB", (64, 64), (10 + idx, 20, 30)).save(image_dir / name)
                writer.writerow([name, "fish", "chair"])
        return root, image_dir, train_csv

    def _run(self, args: list[str]) -> int:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return main(args)


if __name__ == "__main__":
    unittest.main()
