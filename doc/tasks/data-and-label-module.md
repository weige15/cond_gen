# Data and Label Module

## Goal

Implement the shared data layer that parses assignment CSVs, validates labels/files, and exposes deterministic animal/object IDs for training and generation.

## Inputs

- `doc/proposal.md`: Use `dataset/train.csv`, `dataset/trainset/`, and `dataset/generate.csv`; keep animal/object conditioning explicit.
- `doc/high-level-design.md`: Data module owns parsing, mapping, validation, and training/generation row exposure.
- `doc/detailed-design.md`: Follow CSV, label mapping, batch, and failure-handling contracts.
- `doc/test-plan.md`: Cover CSV schema, row counts, labels, duplicates, missing images, and image loading fixtures.

## Write Scope

Shared data helper module, likely under `scripts/`; module-level tests or assertion fixtures; no generated images or checkpoints.

## Read Scope

`dataset/train.csv`, `dataset/generate.csv`, `dataset/trainset/`, `doc/detailed-design.md`, `doc/test-plan.md`.

## Dependencies

None, except the fixed assignment label sets and chosen image library.

## Tasks

- [x] Define fixed animal/object label order and deterministic label maps.
- [x] Implement train/generate CSV loaders with required-column, duplicate-ID, empty-field, and unknown-label checks.
- [x] Implement training image lookup/loading as RGB 64x64 tensors or document the strict resize/verify policy.
- [x] Add fixture checks for valid rows, missing columns, duplicate IDs, unknown labels, missing images, and stable label mappings.
- [x] Add a full-data check path for 4,799 train rows, 2,000 generate rows, expected labels, and referenced training files.

## Tests and Quality Gates

- [x] Data checks pass for tiny fixtures and full assignment CSV/file presence.
- [x] Project data test command is `python -m unittest discover -s tests -p 'test_*.py'`.

## Done When

- [x] Training and generation rows can be loaded and validated independently of model code.
- [x] Invalid CSV, label, duplicate, or missing-image cases fail clearly.
- [x] Module checks cover the Data and Label rows in `doc/test-plan.md`.

## Evidence

- Implemented `scripts/data.py` with CSV loaders, deterministic label maps, strict 64x64 RGB tensor loading, assignment-data validation, and a small data-check CLI.
- Added `tests/test_data.py` fixture checks.
- Passed: `python -m unittest discover -s tests -p 'test_*.py'` (8 tests).
- Passed: `python scripts/data.py --train_csv dataset/train.csv --generate_csv dataset/generate.csv --image_dir dataset/trainset --full --decode_images`.
