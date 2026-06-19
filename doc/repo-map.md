# Repository Map

## Repository Summary

This repository is a sparse HW6 assignment workspace for conditional Brainrot image generation. It currently contains assignment documentation, local dataset/scoring assets, and generated planning documentation, but no model implementation, training script, generation script, or environment manifest.

Tracked source-like content is limited to repository docs and assignment context. The local dataset, scoring program, reference archives, scoring manual, and Windows `Zone.Identifier` sidecar files are present but ignored by `.gitignore`.

## Directory Structure

- `./` - repository root.
- `doc/` - untracked generated documentation; currently includes `problem-brief.md` and this repository map.
- `dataset/` - ignored assignment dataset directory.
- `dataset/trainset/` - ignored training images; 4,799 PNG files observed.
- `scoring_program/` - ignored local scoring program directory.
- `scoring_program/input/ref/` - local scoring reference statistics and config.
- `scoring_program/input/res/` - local scoring results directory; no files observed.
- `.agents/` and `.codex/` - read-only local agent directories; no files found at max depth 2.
- `.git/` - valid git repository metadata.

## Main Source Files

- `scoring_program/score.py` - Python CLI for local image-generation scoring. It defines FID and CLIP scoring helpers, image-size validation, dataset wrappers, and a `main()` entry point.
- No generator, model, dataset loader, training loop, sampling code, or package source directory was discovered.

## Existing Tests

- No test directory, `test_*.py`, unit test config, or test command was discovered.
- `scoring_program/score.py` is an evaluator, not a project test suite.

## Build System

- No `pyproject.toml`, `setup.py`, `requirements*.txt`, `package.json`, lockfile, `Makefile`, `Dockerfile`, `Cargo.toml`, `go.mod`, Maven, or Gradle config was found within the first three directory levels.
- Package manager and reproducible environment are currently unspecified.

## Runtime or CLI Entry Points

- Confirmed evaluator entry point: `scoring_program/score.py`.
- `scoring_program/metadata` declares: `python3 score.py --input_dir $input --output_dir $output --config config.json`.
- `scoring_manual.txt` gives a local scoring example:

```bash
cd scoring_program
python score.py --input_dir ./input --output_dir ./ --image_size 64 --num_images 3000 --test_json test.json --score fid clip_t clip_i --verbose
```

- No training or image-generation CLI entry point exists yet.

## Data and Assets

- `HW6-Brainrot Image Generation.pdf` - assignment PDF, tracked.
- `dataset/train.csv` - ignored CSV with 4,799 data rows plus header.
- `dataset/generate.csv` - ignored CSV with 2,000 data rows plus header.
- `dataset/trainset/` - ignored training image set with 4,799 PNG files.
- `dataset/trainset/` also contains 4,799 `Zone.Identifier` sidecar files.
- `scoring_program/input/ref/test_mu.npy` and `test_sigma.npy` - reference statistics for local FID scoring.
- `scoring_program/input/ref/config.json` - scorer config with `image_size: 64`, `num_images: 3000`, and scores `fid`, `clip_t`, `clip_i`.
- `scoring_program/input/ref/test.json` and `scoring_program/input/ref/test/` were not observed.
- `scoring_program/input/res/` exists and is empty.
- `scoring_program.zip` and `hw6_reference.zip` are present and ignored.

## Existing Documentation

- `README.md` - tracked, currently only contains `# cond_gen`.
- `AGENTS.md` - tracked repository guardrails and assignment operating rules.
- `doc/problem-brief.md` - untracked generated problem brief from the assignment PDF and local files.
- `conditional_score_diffusion_heuristics.md` - tracked note with conceptual diffusion heuristics and compliance boundaries.
- `scoring_manual.txt` - ignored local scoring instructions.
- `HW6-Brainrot Image Generation.pdf` - tracked assignment source.

## Detected Dependencies

No dependency manifest was found. Dependencies inferred only from `scoring_program/score.py` imports:

- Python standard library: `argparse`, `glob`, `json`, `os`, `collections`, `typing`.
- Third-party Python packages: `numpy`, `open_clip`, `torch`, `PIL`, `torchvision`, `tqdm`, `scipy`.

## Important Scripts

- `scoring_program/score.py` - local evaluator for FID, CLIP-T, and CLIP-I.
- No project-owned training, generation, validation, lint, format, build, or packaging scripts were discovered.

## Current Git State

- Current branch: `main`.
- Worktree: `/home/kuotzuwei15/GenAI/hw6` at commit `e34beb7`.
- Remote: `origin git@github.com:weige15/cond_gen.git`.
- Recent commits:
  - `e34beb7 added assignment pdf, and AGENTS.md`
  - `0f2c6b9 first commit`
- Tracked files from `git ls-files`: `.gitignore`, `AGENTS.md`, `HW6-Brainrot Image Generation.pdf`, `README.md`, `conditional_score_diffusion_heuristics.md`.
- Untracked: `doc/`.
- Ignored but present: `dataset/`, `scoring_program/`, `scoring_manual.txt`, `scoring_program.zip`, `hw6_reference.zip`, and `*:Zone.Identifier` files.

## Missing or Ambiguous Areas

- Main implementation is absent: no model code, training command, generation command, checkpoint, or generated final images.
- Reproducible environment is absent: no requirements file or package metadata.
- Local scoring reference appears partial for CLIP scoring: `test_mu.npy`, `test_sigma.npy`, and `config.json` exist, but `test.json` and reference test images were not found.
- Local scorer config/manual use `num_images: 3000`, while `dataset/generate.csv` contains 2,000 generation requests.
- `README.md` does not yet describe training, generation, environment setup, checkpoint use, or package versions.
- No tests or lightweight validation script exist for generated image count, filenames, RGB mode, or 64x64 size.

## Notes for Future Skills

- Use `doc/problem-brief.md` as the assignment-grounded source before proposal or implementation work.
- Preserve `.gitignore` intent: keep datasets, scoring assets, archives, checkpoints, and generated image folders out of source changes unless explicitly requested.
- Future implementation should add the smallest reproducible Python project surface needed for training, generation, validation, and README instructions.
- Resolve the 2,000-vs-3,000 scoring mismatch before treating local evaluator output as submission-equivalent.
