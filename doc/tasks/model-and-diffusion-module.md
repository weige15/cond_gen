# Model and Diffusion Module

## Goal

Implement the from-scratch conditional DDPM model and diffusion utilities needed by training and generation.

## Inputs

- `doc/proposal.md`: Small label-conditioned DDPM with from-scratch UNet-style denoiser.
- `doc/high-level-design.md`: Model/diffusion module owns denoiser, conditioning, schedule, loss, sampler, and checkpoint-relevant config.
- `doc/detailed-design.md`: Follow model config, diffusion config, sampling config, loss, and sampler contracts.
- `doc/test-plan.md`: Cover tensor shapes, finite loss, diffusion schedule sanity, sampler shape/range, and backpropagation.

## Write Scope

Shared model/diffusion helper module, likely `scripts/model.py`; focused model tests or assertion fixtures; no pretrained generator code.

## Read Scope

`doc/detailed-design.md`, `conditional_score_diffusion_heuristics.md`, and any implemented data/checkpoint helpers.

## Dependencies

Data and Label Module for label counts/mappings; chosen Python/PyTorch environment.

## Tasks

- [x] Define minimal model, diffusion, and sampling config structures used in checkpoints.
- [x] Implement a compact conditional UNet-style denoiser trained from scratch.
- [x] Implement timestep, animal, and object conditioning injection.
- [x] Implement DDPM forward noising, MSE noise-prediction loss, and reverse sampling.
- [x] Add finite-loss, forward-shape, schedule-sanity, sampler-shape, and backpropagation checks on tiny tensors.

## Tests and Quality Gates

- [x] Tiny tensor checks pass without training on the full dataset.
- [x] No pretrained generator weights, `diffusers` pipeline, or high-level generation framework is introduced.

## Done When

- [x] Training CLI can call a scalar finite loss.
- [x] Generation CLI can call a sampler that returns RGB-shaped 64x64 tensors.
- [x] Module checks cover the Model and Diffusion rows in `doc/test-plan.md`.

## Evidence

- Implemented `scripts/model.py` with `ModelConfig`, `DiffusionConfig`, a compact from-scratch conditional UNet, DDPM forward loss, reverse sampler, and image tensor conversion helper.
- Added `tests/test_model.py` tensor checks.
- Passed: `python -m unittest tests.test_model` (4 tests).
- Passed: `python -m unittest discover -s tests -p 'test_*.py'` (12 tests).

## Architecture Revision Evidence

- Upgraded `scripts/model.py` from the compact baseline to a configurable A100-oriented conditional UNet with channel multipliers, residual block depth, FiLM-style conditioning, animal/object/pair embeddings, classifier-free condition dropout, low-resolution self-attention, cosine or linear schedules, DDIM-style reduced-step sampling, and classifier-free guidance.
- Passed: `python -m unittest tests.test_model` (5 tests).
- Passed: `python -m unittest discover -s tests -p 'test_*.py'` (31 tests).
