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

- [x] Implement CLI arguments for generation CSV, image directory, image size, and expected count.
- [x] Read expected IDs and reject duplicate IDs.
- [x] Compare expected and actual filename sets and report missing/extra files.
- [x] Verify every expected file is PNG, RGB, and exactly 64x64.
- [x] Return nonzero on any validation failure and print actionable error examples.
- [x] Add fixtures for valid output, missing file, extra file, duplicate ID, grayscale/RGBA image, wrong size, and non-PNG data.

## Tests and Quality Gates

- [x] Validator fixture checks pass.
- [x] Final `generated_images/` must pass this command before scoring or upload.

## Done When

- [x] Validator catches all output-contract failures listed in `doc/test-plan.md`.
- [x] Validator can be run independently of training and model code.
- [x] README can document the validator as the first required post-generation gate.

## Evidence

- Implemented `scripts/validate_generated.py` with exact filename-set, count, PNG, RGB, and image-size checks.
- Added `tests/test_validate_generated.py` fixtures for valid output, missing file, extra file, duplicate ID, grayscale/RGBA modes, wrong size, and non-PNG data.
- Passed: `python -m unittest tests.test_validate_generated` (7 tests).
- Passed: `python -m unittest discover -s tests -p 'test_*.py'` (22 tests).
- Passed direct CLI fixture: `python scripts/validate_generated.py --generate_csv ... --image_dir ... --expected_count 1`.
