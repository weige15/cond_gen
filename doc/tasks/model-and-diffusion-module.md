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

- [ ] Define minimal model, diffusion, and sampling config structures used in checkpoints.
- [ ] Implement a compact conditional UNet-style denoiser trained from scratch.
- [ ] Implement timestep, animal, and object conditioning injection.
- [ ] Implement DDPM forward noising, MSE noise-prediction loss, and reverse sampling.
- [ ] Add finite-loss, forward-shape, schedule-sanity, sampler-shape, and backpropagation checks on tiny tensors.

## Tests and Quality Gates

- [ ] Tiny tensor checks pass without training on the full dataset.
- [ ] No pretrained generator weights, `diffusers` pipeline, or high-level generation framework is introduced.

## Done When

- [ ] Training CLI can call a scalar finite loss.
- [ ] Generation CLI can call a sampler that returns RGB-shaped 64x64 tensors.
- [ ] Module checks cover the Model and Diffusion rows in `doc/test-plan.md`.
