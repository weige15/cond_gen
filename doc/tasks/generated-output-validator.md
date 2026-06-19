# Generated Output Validator

## Goal

Implement the fast project-owned quality gate for generated image count, filenames, PNG format, RGB mode, and 64x64 size.

## Inputs

- `doc/proposal.md`: Validation is required before scoring or submission.
- `doc/high-level-design.md`: Validator owns deterministic output-contract checks.
- `doc/detailed-design.md`: Follow validator command, exact-set comparison, and image metadata contracts.
- `doc/test-plan.md`: Cover missing files, extra files, duplicate IDs, wrong mode, wrong size, and non-PNG cases.

## Write Scope

`scripts/validate_generated.py`; validator fixture tests/checks; README validation command snippet after command is stable.

## Read Scope

`dataset/generate.csv`, generated image directory, `doc/detailed-design.md`, `doc/test-plan.md`.

## Dependencies

Data and Label Module may be reused for generation CSV parsing; image reader dependency.

## Tasks

- [ ] Implement CLI arguments for generation CSV, image directory, image size, and expected count.
- [ ] Read expected IDs and reject duplicate IDs.
- [ ] Compare expected and actual filename sets and report missing/extra files.
- [ ] Verify every expected file is PNG, RGB, and exactly 64x64.
- [ ] Return nonzero on any validation failure and print actionable error examples.
- [ ] Add fixtures for valid output, missing file, extra file, duplicate ID, grayscale/RGBA image, wrong size, and non-PNG data.

## Tests and Quality Gates

- [ ] Validator fixture checks pass.
- [ ] Final `generated_images/` must pass this command before scoring or upload.

## Done When

- [ ] Validator catches all output-contract failures listed in `doc/test-plan.md`.
- [ ] Validator can be run independently of training and model code.
- [ ] README can document the validator as the first required post-generation gate.
