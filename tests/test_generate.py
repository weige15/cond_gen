from __future__ import annotations

import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.checkpoint import CHECKPOINT_VERSION, label_maps_to_dict, load_checkpoint, save_checkpoint
from scripts.data import GenerateRow, build_label_maps
from scripts.generate import _parse_devices, _shard_rows, main
from scripts.model import DiffusionConfig, ModelConfig, build_model, diffusion_config_to_dict, model_config_to_dict
from scripts.validate_generated import main as validate_main


class GenerationCliTests(unittest.TestCase):
    def test_missing_checkpoint_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generate_csv = self._generate_csv(root, ["000001.png"])
            code = self._run(["--checkpoint", str(root / "missing.pth"), "--generate_csv", str(generate_csv)])
            self.assertNotEqual(code, 0)

    def test_tiny_generation_writes_exact_files_and_passes_validator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = self._checkpoint(root)
            generate_csv = self._generate_csv(root, ["000001.png", "000002.png"])
            output_dir = root / "generated"
            code = self._run(
                [
                    "--checkpoint",
                    str(checkpoint),
                    "--generate_csv",
                    str(generate_csv),
                    "--output_dir",
                    str(output_dir),
                    "--batch_size",
                    "1",
                    "--device",
                    "cpu",
                    "--expected_count",
                    "2",
                    "--sampling_steps",
                    "2",
                ]
            )
            self.assertEqual(code, 0)
            self.assertEqual({p.name for p in output_dir.iterdir()}, {"000001.png", "000002.png"})
            self.assertEqual(
                self._validate(generate_csv, output_dir, 2),
                0,
            )

    def test_duplicate_generation_id_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = self._checkpoint(root)
            generate_csv = self._generate_csv(root, ["000001.png", "000001.png"])
            code = self._run(
                [
                    "--checkpoint",
                    str(checkpoint),
                    "--generate_csv",
                    str(generate_csv),
                    "--output_dir",
                    str(root / "generated"),
                    "--expected_count",
                    "2",
                ]
            )
            self.assertNotEqual(code, 0)

    def test_incompatible_checkpoint_mapping_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = self._checkpoint(root)
            payload = load_checkpoint(checkpoint)
            payload["label_maps"]["animal_to_id"]["fish"] = 99
            torch.save(payload, checkpoint)
            generate_csv = self._generate_csv(root, ["000001.png"])
            code = self._run(
                [
                    "--checkpoint",
                    str(checkpoint),
                    "--generate_csv",
                    str(generate_csv),
                    "--output_dir",
                    str(root / "generated"),
                    "--expected_count",
                    "1",
                ]
            )
            self.assertNotEqual(code, 0)

    def test_stale_output_requires_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = self._checkpoint(root)
            generate_csv = self._generate_csv(root, ["000001.png"])
            output_dir = root / "generated"
            output_dir.mkdir()
            (output_dir / "000001.png").write_text("stale", encoding="utf-8")
            code = self._run(
                [
                    "--checkpoint",
                    str(checkpoint),
                    "--generate_csv",
                    str(generate_csv),
                    "--output_dir",
                    str(output_dir),
                    "--expected_count",
                    "1",
                ]
            )
            self.assertNotEqual(code, 0)

    def test_multi_device_helpers_parse_and_shard_rows(self) -> None:
        rows = [
            GenerateRow(f"{index:06d}.png", "fish", "chair", "a fish and a chair", 8, 9)
            for index in range(5)
        ]

        self.assertEqual(_parse_devices("0, cuda:1"), ["cuda:0", "cuda:1"])
        self.assertEqual(
            [[row.image_id for row in shard] for shard in _shard_rows(rows, 2)],
            [["000000.png", "000002.png", "000004.png"], ["000001.png", "000003.png"]],
        )

    def _checkpoint(self, root: Path) -> Path:
        label_maps = build_label_maps()
        model_config = ModelConfig(
            base_channels=4,
            channel_mults=(1, 2),
            num_res_blocks=1,
            attention_resolutions=(),
            cond_dim=32,
            dropout=0.0,
            condition_dropout=0.0,
        )
        diffusion_config = DiffusionConfig(timesteps=2, schedule="linear")
        model = build_model(model_config)
        checkpoint = root / "model.pth"
        save_checkpoint(
            {
                "format_version": CHECKPOINT_VERSION,
                "model_state": model.state_dict(),
                "model_config": model_config_to_dict(model_config),
                "diffusion_config": diffusion_config_to_dict(diffusion_config),
                "label_maps": label_maps_to_dict(label_maps),
                "seed": 0,
                "training": {"steps": 0},
            },
            checkpoint,
        )
        return checkpoint

    def _generate_csv(self, root: Path, names: list[str]) -> Path:
        path = root / "generate.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "animal", "object", "prompt"])
            for name in names:
                writer.writerow([name, "fish", "chair", "a fish and a chair"])
        return path

    def _run(self, args: list[str]) -> int:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return main(args)

    def _validate(self, generate_csv: Path, output_dir: Path, expected_count: int) -> int:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return validate_main(
                [
                    "--generate_csv",
                    str(generate_csv),
                    "--image_dir",
                    str(output_dir),
                    "--expected_count",
                    str(expected_count),
                ]
            )


if __name__ == "__main__":
    unittest.main()
