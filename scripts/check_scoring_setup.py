from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check local scorer inputs without running the scorer.")
    parser.add_argument("--scoring_dir", type=Path, default=Path("scoring_program"))
    parser.add_argument("--generated_dir", type=Path, default=Path("generated_images"))
    parser.add_argument("--expected_count", type=int, default=2000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    errors, warnings = check_scoring_setup(args.scoring_dir, args.generated_dir, args.expected_count)
    for warning in warnings:
        print(f"warning: {warning}")
    if errors:
        print("local scoring setup incomplete:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("local scoring setup files are present")
    return 0


def check_scoring_setup(scoring_dir: Path, generated_dir: Path, expected_count: int) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    ref_dir = scoring_dir / "input" / "ref"

    for path in (
        scoring_dir / "score.py",
        ref_dir / "test_mu.npy",
        ref_dir / "test_sigma.npy",
        ref_dir / "test.json",
    ):
        if not path.is_file():
            errors.append(f"missing file: {path}")

    if not (ref_dir / "test").is_dir():
        errors.append(f"missing reference image directory: {ref_dir / 'test'}")

    if not generated_dir.is_dir():
        errors.append(f"missing generated result directory: {generated_dir}")
    else:
        count = len([path for path in generated_dir.iterdir() if path.is_file()])
        if count != expected_count:
            errors.append(f"generated result count mismatch in {generated_dir}: expected {expected_count}, got {count}")

    config_path = ref_dir / "config.json"
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        scorer_count = config.get("num_images")
        if scorer_count is not None and scorer_count != expected_count:
            warnings.append(f"scorer config num_images={scorer_count}, expected generated count={expected_count}")

    return errors, warnings


if __name__ == "__main__":
    raise SystemExit(main())
