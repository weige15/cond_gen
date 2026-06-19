# Test Plan

## Purpose

This plan defines verification criteria before detailed implementation starts. "Done" for verification means each HLD module has an observable check, generated outputs are validated by exact pass/fail rules, and known evaluator commands are documented without claiming they were run.

The plan does not implement tests, train a model, generate images, or run the provided scorer.

## Source Requirements

Sources read:

- `doc/proposal.md`: objective, DDPM approach, module candidates, validation plan, risks, assumptions, open questions.
- `doc/high-level-design.md`: modules, relationships, data flow, interfaces, contracts, quality-gate alignment.
- `doc/problem-brief.md`: assignment requirements, dataset counts, constraints, scoring bands, deliverables.
- `doc/repo-map.md`: current repository state and missing scripts/tests/environment.
- `doc/quality-gates.md`: discovered evaluator command, missing quality gates, recommended done criteria.

Extracted requirements:

- Train the main conditional generator from scratch.
- Use `dataset/train.csv`, `dataset/trainset/`, and `dataset/generate.csv`.
- Generate exactly 2,000 RGB PNG images at 64x64.
- Match generated filenames exactly to `dataset/generate.csv`.
- Preserve animal/object conditioning.
- Keep generated images, checkpoints, datasets, and scorer outputs out of source control unless packaging is explicitly requested.
- Document environment, training command, generation command, validation command, checkpoint, seed/config, and any extra data.

Missing or ambiguous source material:

- Python, CUDA, PyTorch, and package-manager versions are unspecified.
- Training hardware and runtime budget are unspecified.
- Local scorer setup is incomplete: `test.json` and reference test images were not observed.
- Local scorer uses 3,000 images, while assignment generation requires 2,000 images.
- No project-owned implementation, tests, or validation script exists yet.

## Test Scope

Covered by this plan:

- Data and Label Module CSV parsing, class mapping, row counts, image existence, and invalid-input handling.
- Model and Diffusion Module tensor contracts, conditioning contract, diffusion schedule consistency, loss shape/value sanity, and sampler output range/shape.
- Training CLI argument handling, checkpoint persistence, seed/config recording, and small overfit workflow.
- Generation CLI checkpoint loading, generation row iteration, output naming, image format, and deterministic seed behavior when supported.
- Generated Output Validator exact output contract.
- Documentation and Environment Manifest reproducibility requirements.
- Optional Local Scoring Adapter command documentation and setup checks.

Noted coverage status:

- Existing: only the external `scoring_program/score.py` evaluator exists; it is not a project test suite.
- Planned: all project-owned tests and validators.
- Missing: test runner command, smoke-test command, implementation modules, and validation script.
- Unknown: final benchmark thresholds for training time, sampling time, and model quality beyond official FID/CLIP-T bands.

## Non-Tested Scope

- Official hidden-test FID and CLIP-T correctness beyond the provided Codabench/evaluator behavior.
- Human aesthetic quality as a sole pass/fail criterion.
- Full leaderboard tuning strategy.
- Extra-data acquisition, licensing, and provenance unless the final solution uses extra data.
- Alternative architectures such as GAN, VAE, Transformer diffusion, or latent diffusion unless the design changes.
- Pretrained auxiliary module behavior except official/local scoring or explicitly approved diagnostics.

## Smoke Tests

Planned smoke checks:

| Check | Status | Pass Criteria | Failure Examples |
| --- | --- | --- | --- |
| Repository data presence | Planned | `dataset/train.csv`, `dataset/generate.csv`, and `dataset/trainset/` exist | Missing CSV, missing image directory |
| CSV row counts | Planned | `train.csv` has 4,799 data rows; `generate.csv` has 2,000 data rows | Header-only CSV, duplicate/missing rows |
| Class coverage | Planned | Observed labels are within the 10 animals and 10 objects from the assignment | Unknown animal/object label |
| Training CLI startup | Missing | Command parses arguments and can run a no-train/data-check mode once implemented | Import failure, missing dependency, invalid defaults |
| Generation CLI startup | Missing | Command parses arguments and rejects missing checkpoint with a clear error | Silent success without checkpoint |
| Validator startup | Missing | Validator can check a tiny fixture directory and exit nonzero on invalid images | Accepts wrong size or wrong filename |

No smoke command is verified. `doc/quality-gates.md` marks smoke-test commands as missing.

## Unit Tests by Module

| HLD Module | Verification Method | Status | Pass Criteria |
| --- | --- | --- | --- |
| Data and Label Module | Unit tests with tiny CSV fixtures for `train.csv` and `generate.csv` schemas | Planned | Required columns are parsed; missing columns fail clearly |
| Data and Label Module | Label mapping tests for all assignment animals/objects | Planned | Known labels map deterministically; unknown labels fail |
| Data and Label Module | Image loading tests with valid and invalid PNG fixtures | Planned | Valid RGB 64x64 image loads; missing/wrong-mode/wrong-size image fails where required |
| Model and Diffusion Module | Tensor-shape tests for denoiser forward contract | Planned | Input noisy batch, timestep, animal/object labels produce predicted-noise tensor with same image shape |
| Model and Diffusion Module | Diffusion schedule sanity tests | Planned | Schedule tensors have expected length, monotonic cumulative alpha behavior, and no NaN/Inf |
| Model and Diffusion Module | Loss sanity test | Planned | MSE loss is scalar, finite, and backpropagates through model parameters |
| Model and Diffusion Module | Sampler shape/range test with tiny step count | Planned | Sampler returns batch of RGB-shaped tensors convertible to 64x64 images |
| Training CLI | Argument and config recording tests | Missing | CLI records seed, dataset paths, model settings, and checkpoint path |
| Training CLI | Checkpoint contract test | Missing | Saved checkpoint contains model weights, label mapping, and sampling settings |
| Generation CLI | Checkpoint-loading contract test | Missing | Generation rejects incompatible/missing checkpoint and accepts valid checkpoint |
| Generation CLI | Output naming test using tiny `generate.csv` fixture | Missing | Writes exactly requested filenames and no extras |
| Generated Output Validator | Full output-contract tests | Missing | Detects missing files, extra files, wrong mode, wrong size, and non-PNG files |
| Documentation and Environment Manifest | Documentation checklist review | Planned | README includes setup, train/generate/validate commands, seed/config, checkpoint, versions, and extra-data note |
| Optional Local Scoring Adapter | Setup validation/documentation check | Missing | Does not claim scoring is runnable unless `test.json`, reference images, generated images, dependencies, and command are present |

## Integration Tests

Planned integration checks:

- Data-to-training dry run: load a tiny fixture dataset, construct label mappings, initialize model, run one optimization step, and save a checkpoint.
- Training-to-generation compatibility: train or initialize a tiny checkpoint, load it through the generation path, and write outputs for a tiny `generate.csv`.
- Generation-to-validation pipeline: generate or fixture-create images, then validate exact filenames, count, PNG format, RGB mode, and 64x64 size.
- Full dataset contract check: parse real assignment CSVs and verify row counts, label set, and referenced training image existence without training.
- README command consistency check: documented command names and paths match implemented scripts.

Status: Missing until implementation exists.

## Golden Test Cases

Planned hand-solvable cases:

| Case | Input | Expected Behavior |
| --- | --- | --- |
| Minimal valid training CSV | Header plus one row: `000001.png,fish,chair`; matching RGB 64x64 PNG exists | Data module returns one sample with animal `fish`, object `chair`, and image tensor |
| Missing training image | CSV references `missing.png`; file absent | Data module fails with a clear missing-file error |
| Unknown class | CSV row uses animal `dragon` | Label validation fails before training |
| Minimal generation CSV | Two rows with IDs `000001.png`, `000002.png` | Generation/validator expects exactly those two filenames in fixture mode |
| Duplicate generation ID | Two rows with same `id` | Validation fails because output contract is ambiguous |
| Wrong image mode | Generated file is grayscale PNG | Validator fails because image is not RGB |
| Wrong image size | Generated file is RGB 32x32 PNG | Validator fails because image is not 64x64 |
| Extra generated file | Directory contains all expected files plus `extra.png` | Validator fails because extra files are present |

Exact generated image pixels are not a golden-test target because the generator is stochastic and quality is evaluated by FID/CLIP-T.

## Oracle or Reference Implementation Strategy

Use simple oracles for deterministic contracts:

- CSV oracle: Python standard CSV parsing against tiny fixture files with exact expected rows and labels.
- Filename oracle: set equality between expected IDs from `generate.csv` and generated directory filenames.
- Image oracle: image metadata checks for PNG format, RGB mode, and 64x64 size.
- Checkpoint oracle: required keys/metadata are present and sufficient for generation.

No brute-force oracle is feasible for final image quality. For stochastic generation, oracle checks cover contract validity; quality remains evaluator/manual-sample based.

## Randomized or Property Tests

Planned property checks:

- CSV order property: shuffled generation rows still produce exactly the same expected filename set.
- Label mapping property: repeated parsing of the same CSV produces stable animal/object ID mappings.
- Validator property: removing any one expected generated file makes validation fail.
- Validator property: adding any extra generated file makes validation fail.
- Sampler shape property: for deterministic test seeds and small batch sizes, sampler output shape is always `(batch, 3, 64, 64)` or the documented equivalent image layout.
- Numerical property: model loss and diffusion schedule outputs are finite for small random tensors and valid labels.

Use deterministic seeds and bounded tiny fixtures. Status: Missing until tests are implemented.

## Edge Cases

Planned edge cases:

- Empty CSV file.
- CSV with header only.
- CSV with missing required columns.
- CSV with duplicate IDs.
- CSV with labels outside assignment label sets.
- CSV with valid labels but missing image files.
- Training image exists but is not PNG.
- Training image is PNG but wrong size or color mode.
- Generated directory missing.
- Generated directory empty.
- Generated directory has correct count but wrong filenames.
- Generated directory has correct filenames but wrong image mode/size.
- Checkpoint missing label mapping.
- Checkpoint created with incompatible model/diffusion settings.
- Generation output directory already contains stale extra files.
- Local scorer reference inputs missing.

## Performance Benchmarks

Known benchmark/evaluation metrics:

- Official FID and CLIP-T score bands are documented in `doc/problem-brief.md`.
- The PDF baseline table mentions RTX 4070 12GB VRAM, but this is not a required local hardware target.

Planned benchmarks:

- Small overfit benchmark: model should overfit a tiny subset enough to demonstrate training/sampling mechanics. Exact image-quality threshold is Unknown.
- Full generation benchmark: generation should complete 2,000 images without crashing. Runtime threshold is Unknown until hardware is known.
- Optional local FID/CLIP-T run: use provided scorer only after local inputs, dependencies, GPU, and 2,000-vs-3,000 scorer question are resolved.

No performance command is verified.

## Evaluator or Grading Commands

Known, not run:

```bash
cd scoring_program
python score.py --input_dir ./input --output_dir ./ --image_size 64 --num_images 3000 --test_json test.json --score fid clip_t clip_i --verbose
```

Reason not run: `doc/quality-gates.md` says it writes `scores.json`, may use GPU/model weights, and local `input/ref/test.json`, `input/ref/test/`, and generated result images are absent.

Known, not directly runnable locally without harness paths:

```bash
python3 score.py --input_dir $input --output_dir $output --config config.json
```

Unknown:

- Project-owned unit test command.
- Project-owned smoke-test command.
- Training command.
- Generation command.
- Validation command.
- Lint/format/type-check commands.

## Regression Tests

Planned regression checks once bugs are found or implementation lands:

- Generated output must not silently skip rows from `generate.csv`.
- Generated output must not overwrite requested filenames with stale images from previous runs without an explicit policy.
- Validator must fail on extra files, not only missing files.
- Validator must reject grayscale/RGBA images if assignment requires RGB.
- Generation must preserve exact filename strings from `generate.csv`.
- Checkpoint loading must fail clearly when label mappings or diffusion settings do not match the generator.
- README commands must be updated whenever script paths or required arguments change.

Status: Missing until implementation and first defect cycle exist.

## Manual Verification

Manual checks required because they are not fully captured by unit tests:

- Inspect per-pair sample grids during training to catch condition failures.
- Confirm generated images visibly contain both the animal and object often enough to justify a Codabench upload.
- Review README for reproducibility clarity.
- Review any external reference/code usage for license/provenance compliance.
- Confirm no checkpoints, generated images, scoring outputs, datasets, or archives are staged accidentally unless packaging is explicitly requested.
- Confirm local scoring setup and 2,000-vs-3,000 behavior before interpreting local scores.

Manual inspection cannot replace the generated-output validator.

## Minimum Done Criteria

Before detailed implementation is considered ready to proceed, the planned design must support these pass/fail gates:

- Every HLD module has at least one planned verification method in this document.
- The implementation plan includes a project-owned generated-output validator.
- The validator contract checks exact count, exact filenames, PNG format, RGB mode, and 64x64 size.
- Data checks cover `train.csv` row count, `generate.csv` row count, label sets, and training image existence.
- Training/generation compatibility is covered by a checkpoint round-trip integration test plan.
- Known evaluator commands are labeled `Known, not run`, not verified.
- Unknown test, train, generate, validate, lint, format, and type-check commands remain marked `Unknown` until implemented.
- Open local scorer issues remain explicit: missing reference inputs and the 2,000-vs-3,000 mismatch.

## Open Questions

- Which Python, CUDA, PyTorch, and package-manager versions should become the supported test environment?
- What command should run project-owned unit/smoke tests once implementation exists?
- Should tests use `pytest`, plain `unittest`, or a no-framework assertion script?
- What GPU/runtime budget should define performance pass/fail thresholds?
- How should local scoring be configured given the assignment requires 2,000 generated images but the scorer config/manual refers to 3,000?
- Should pretrained CLIP be used only through the official/local scorer, or also in optional diagnostics?
- Will extra data be used, requiring additional provenance and reproducibility checks?
