# Generation CLI

## Goal

Implement the user-facing generation command that loads a checkpoint and writes one PNG for every row in `dataset/generate.csv`.

## Inputs

- `doc/proposal.md`: Generate exactly 2,000 RGB PNGs at 64x64 using filenames from `generate.csv`.
- `doc/high-level-design.md`: Generation CLI owns requested output loop and output filenames.
- `doc/detailed-design.md`: Follow generation config, checkpoint compatibility, stale-output policy, and image-save contracts.
- `doc/test-plan.md`: Cover missing checkpoint, incompatible mapping, duplicate IDs, exact filenames, and generation-to-validator flow.

## Write Scope

`scripts/generate.py`; generation integration checks; README generation command snippet after command is stable.

## Read Scope

Training checkpoint contract, Data and Label Module, Model and Diffusion Module, `dataset/generate.csv`, `doc/detailed-design.md`.

## Dependencies

Data and Label Module; Model and Diffusion Module; Training CLI checkpoint format.

## Tasks

- [x] Implement CLI argument parsing for checkpoint, generate CSV, output directory, seed, device, batch size, and sampling settings.
- [x] Load and validate checkpoint metadata before sampling.
- [x] Load generation rows and reject duplicate IDs or checkpoint-incompatible labels.
- [x] Sample images in batches and save exact `output_dir/{id}` RGB PNG filenames.
- [x] Define and implement stale-output behavior: require empty output directory or explicit overwrite flag.
- [x] Add tiny fixture generation check that writes exact filenames and passes validator fixtures.

## Tests and Quality Gates

- [x] Missing/incompatible checkpoint checks fail clearly.
- [x] Tiny generation fixture writes no missing or extra filenames.
- [x] Generated files pass the Generated Output Validator in fixture mode.

## Done When

- [x] Generation command can produce requested filenames from a valid tiny checkpoint.
- [x] Output naming and checkpoint compatibility are verified.
- [x] Full 2,000-image generation remains a later run gate, not required for task-file completion.

## Evidence

- Implemented `scripts/generate.py` with checkpoint loading, mapping compatibility checks, stale-output checks, batch sampling, and exact RGB PNG filename writes.
- Added `tests/test_generate.py` covering missing checkpoint, incompatible label maps, duplicate generation IDs, stale output, and generation-to-validator fixture flow.
- Passed: `python -m unittest tests.test_generate` (5 tests).
- Passed: `python -m unittest discover -s tests -p 'test_*.py'` (27 tests).
- Verified direct missing-checkpoint failure: `python scripts/generate.py --checkpoint /tmp/hw6-missing-checkpoint.pth --generate_csv dataset/generate.csv --output_dir /tmp/hw6-no-output --expected_count 2000`.

## Architecture Revision Evidence

- Added EMA checkpoint loading, reduced-step DDIM sampling, classifier-free guidance scale, and DDIM eta controls.
- Passed compact checkpoint-to-generation-to-validator smoke using `--use_ema` and `--sampling_steps 2`.
