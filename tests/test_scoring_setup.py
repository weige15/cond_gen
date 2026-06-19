from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts.check_scoring_setup import check_scoring_setup, main


class ScoringSetupTests(unittest.TestCase):
    def test_missing_reference_inputs_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scoring_dir = root / "scoring_program"
            (scoring_dir / "input" / "ref").mkdir(parents=True)
            (scoring_dir / "score.py").write_text("# scorer\n", encoding="utf-8")
            generated_dir = root / "generated_images"
            generated_dir.mkdir()

            errors, _ = check_scoring_setup(scoring_dir, generated_dir, expected_count=1)
            self.assertTrue(any("test.json" in error for error in errors))
            self.assertTrue(any("test_mu.npy" in error for error in errors))

    def test_complete_fixture_passes_with_count_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scoring_dir = root / "scoring_program"
            ref_dir = scoring_dir / "input" / "ref"
            (ref_dir / "test").mkdir(parents=True)
            (scoring_dir / "score.py").write_text("# scorer\n", encoding="utf-8")
            for name in ("test_mu.npy", "test_sigma.npy", "test.json"):
                (ref_dir / name).write_text("x", encoding="utf-8")
            (ref_dir / "config.json").write_text(json.dumps({"num_images": 3000}), encoding="utf-8")
            generated_dir = root / "generated_images"
            generated_dir.mkdir()
            (generated_dir / "000001.png").write_text("x", encoding="utf-8")

            errors, warnings = check_scoring_setup(scoring_dir, generated_dir, expected_count=1)
            self.assertEqual(errors, [])
            self.assertTrue(any("num_images=3000" in warning for warning in warnings))

    def test_cli_returns_nonzero_when_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = main(["--scoring_dir", str(Path(tmp) / "missing")])
            self.assertNotEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
