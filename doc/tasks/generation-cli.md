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

- [ ] Implement CLI argument parsing for checkpoint, generate CSV, output directory, seed, device, batch size, and sampling settings.
- [ ] Load and validate checkpoint metadata before sampling.
- [ ] Load generation rows and reject duplicate IDs or checkpoint-incompatible labels.
- [ ] Sample images in batches and save exact `output_dir/{id}` RGB PNG filenames.
- [ ] Define and implement stale-output behavior: require empty output directory or explicit overwrite flag.
- [ ] Add tiny fixture generation check that writes exact filenames and passes validator fixtures.

## Tests and Quality Gates

- [ ] Missing/incompatible checkpoint checks fail clearly.
- [ ] Tiny generation fixture writes no missing or extra filenames.
- [ ] Generated files pass the Generated Output Validator in fixture mode.

## Done When

- [ ] Generation command can produce requested filenames from a valid tiny checkpoint.
- [ ] Output naming and checkpoint compatibility are verified.
- [ ] Full 2,000-image generation remains a later run gate, not required for task-file completion.
