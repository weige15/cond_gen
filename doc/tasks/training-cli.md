# Training CLI

## Goal

Implement the user-facing training command that validates data, trains the DDPM, and saves a reproducible checkpoint.

## Inputs

- `doc/proposal.md`: Train from scratch, support small overfit, full training, checkpoint, and README reproducibility.
- `doc/high-level-design.md`: Training CLI owns optimization, checkpoint creation, and documented training command.
- `doc/detailed-design.md`: Follow training command, checkpoint metadata, failure handling, and small-overfit contracts.
- `doc/test-plan.md`: Cover parser startup, data-check path, one-step dry run, checkpoint contract, and missing-path failures.

## Write Scope

`scripts/train.py`; shared checkpoint helper if needed; training smoke/integration checks; README command snippet after command is stable.

## Read Scope

Data and Label Module, Model and Diffusion Module, `dataset/train.csv`, `dataset/trainset/`, `doc/detailed-design.md`.

## Dependencies

Data and Label Module; Model and Diffusion Module; chosen device/environment.

## Tasks

- [ ] Implement CLI argument parsing for train CSV, image directory, checkpoint path, seed, device, and core training settings.
- [ ] Add a data-check-only path that validates assignment inputs without training.
- [ ] Wire dataset, model, diffusion loss, optimizer, seed handling, and training loop.
- [ ] Save checkpoints with model weights, configs, label maps, seed, training settings, and optional EMA metadata.
- [ ] Add tiny fixture dry run that performs one optimization step and verifies required checkpoint keys.

## Tests and Quality Gates

- [ ] Training CLI startup and missing-path checks pass.
- [ ] Tiny dry run saves a checkpoint consumable by Generation CLI.
- [ ] Full training is not required for module completion unless explicitly requested.

## Done When

- [ ] A documented training command exists and can validate data or run a tiny training path.
- [ ] Checkpoints satisfy the shared checkpoint contract.
- [ ] Training-related test-plan checks are covered or explicitly blocked by environment.
