# Detailed Design

## Purpose

This document defines the implementation-level design for the HW6 Brainrot conditional image generation solution without writing code. It preserves the HLD module boundaries, turns the proposal and test plan into concrete interfaces and data contracts, and keeps unresolved environment, scorer, and hyperparameter choices explicit.

## Source Proposal Summary

The proposal targets a reproducible from-scratch conditional image generation pipeline that trains on `dataset/train.csv` and `dataset/trainset/`, then generates exactly 2,000 RGB PNG images at 64x64 from `dataset/generate.csv`.

The chosen baseline is a small label-conditioned DDPM with a from-scratch UNet-style denoiser. Animal and object labels remain explicit conditions. The proposal rejects pretrained generator weights, high-level generation pipelines, and copied external repository code.

## HLD Summary

The HLD defines these modules:

- Data and Label Module
- Model and Diffusion Module
- Training CLI
- Generation CLI
- Generated Output Validator
- Documentation and Environment Manifest
- Optional Local Scoring Adapter

The workflow is:

1. Validate and load assignment CSV/image data.
2. Train a from-scratch conditional DDPM.
3. Save a checkpoint with enough metadata for generation.
4. Generate `generated_images/{id}` for every row in `dataset/generate.csv`.
5. Validate generated filenames, count, PNG format, RGB mode, and 64x64 size.
6. Run local scoring only after scorer setup is resolved and approved.

## Design Goals

- Keep the implementation small enough for an HW6 assignment.
- Make data contracts explicit and fail early on invalid assignment inputs.
- Keep training and generation reproducible through checkpoint metadata and README commands.
- Make generated-output validation independent of model quality.
- Support a tiny fixture path for tests without requiring full training.
- Avoid hidden shared state between scripts.

## Non-Goals

- No full experiment framework.
- No multiple model-family plugin system.
- No external pretrained generator or high-level diffusion pipeline.
- No official-score guarantee from local tests.
- No automated Codabench upload.
- No local scorer rewrite.

## Architecture Overview

The design uses a small Python/PyTorch project surface:

- `scripts/train.py`: training entry point.
- `scripts/generate.py`: generation entry point.
- `scripts/validate_generated.py`: output-contract validator.
- Shared implementation module(s): data loading, label mapping, model, diffusion, checkpoint helpers.
- `requirements.txt`: minimal dependency manifest once versions are chosen.
- `README.md`: reproducibility instructions.

The exact shared module file layout may be refined during implementation, but responsibilities must stay aligned to the HLD modules.

## Shared Data Contracts

### Assignment Label Sets

Animal labels:

```text
shark, crocodile, frog, cat, dog, capybara, elephant, bird, fish, monkey
```

Object labels:

```text
sneaker, airplane, coffee cup, banana, cactus, toilet, pizza, drum, car, chair
```

### Training CSV Contract

Path: `dataset/train.csv`

Required columns:

```text
id,animal,object
```

Required behavior:

- `id` is a PNG filename relative to `dataset/trainset/`.
- `animal` must be in the assignment animal set.
- `object` must be in the assignment object set.
- Expected full assignment row count is 4,799 data rows.
- Duplicate IDs are invalid for full training checks.

### Generation CSV Contract

Path: `dataset/generate.csv`

Required columns:

```text
id,animal,object,prompt
```

Required behavior:

- `id` is the exact output filename.
- `animal` and `object` must be known labels.
- `prompt` is retained for documentation/scoring compatibility, but label conditioning uses `animal` and `object`.
- Expected full assignment row count is 2,000 data rows.
- Duplicate IDs are invalid.

### Image Contract

Training input images:

- File exists at `dataset/trainset/{id}`.
- Decodable image.
- Converted or loaded as RGB for training.
- Resized or verified to 64x64 according to the final implementation choice.

Generated output images:

- File exists at `generated_images/{id}`.
- PNG format.
- RGB mode.
- Exact size 64x64.
- Filename set exactly equals IDs in `dataset/generate.csv`.

### Label Mapping Contract

Use deterministic mappings based on the fixed assignment label order above.

Checkpoint metadata must store either:

- the exact `animal_to_id` and `object_to_id` mappings, or
- a versioned reference to the fixed assignment label order.

Generation must fail if checkpoint mappings are incompatible with generation labels.

### Batch Contract

Training batch fields:

- `image`: normalized RGB image tensor, shape `(batch, 3, 64, 64)`.
- `animal_id`: integer tensor, shape `(batch,)`.
- `object_id`: integer tensor, shape `(batch,)`.
- `image_id`: filename strings for trace/debug only.

Generation request fields:

- `image_id`: output filename.
- `animal_id`: integer label.
- `object_id`: integer label.
- `prompt`: original prompt string, retained for traceability.

### Checkpoint Contract

Checkpoint must include:

- model weights.
- model configuration needed to rebuild the architecture.
- diffusion configuration needed to sample.
- label mappings.
- training seed and important training settings.
- optional EMA weights if EMA is implemented.

Generation must reject checkpoints missing required generation metadata.

## Module Designs

### Data and Label Module

#### Responsibility

Own CSV parsing, label validation, deterministic label mappings, training image lookup, generation-row loading, and lightweight dataset validation.

#### Non-Responsibility

- No model construction.
- No optimizer or training loop.
- No image generation.
- No scoring.
- No submission packaging.

#### Inputs and Outputs

Inputs:

- `dataset/train.csv`
- `dataset/generate.csv`
- `dataset/trainset/`
- fixed assignment label sets

Outputs:

- validated training rows.
- validated generation rows.
- label mappings.
- training samples for the Training CLI.

#### Public Interface

Design-level interface:

- `load_train_rows(train_csv_path) -> list[TrainRow]`
- `load_generate_rows(generate_csv_path) -> list[GenerateRow]`
- `build_label_maps() -> LabelMaps`
- `validate_train_rows(rows, image_dir, label_maps, expected_count=None) -> ValidationResult`
- `validate_generate_rows(rows, label_maps, expected_count=None) -> ValidationResult`
- `BrainrotDataset(rows, image_dir, label_maps) -> dataset-like object`

The implementation may choose exact names, but these capabilities must remain independently testable.

#### Data Structures

- `TrainRow`: `image_id`, `animal`, `object`.
- `GenerateRow`: `image_id`, `animal`, `object`, `prompt`.
- `LabelMaps`: `animal_to_id`, `id_to_animal`, `object_to_id`, `id_to_object`.
- `ValidationResult`: pass/fail plus error messages, or fail-fast exceptions with clear messages.

#### Internal Design

- Use standard CSV parsing.
- Normalize only structural whitespace around CSV fields if present; do not rewrite label spellings.
- Validate required columns before reading rows.
- Validate label membership before training/generation.
- For full assignment checks, compare row counts to 4,799 and 2,000.
- For fixture tests, allow caller-provided expected counts or no full-count enforcement.

#### Algorithm Details

CSV loading:

1. Open CSV.
2. Confirm required header columns exist.
3. Read rows in file order.
4. Reject empty `id`, `animal`, or `object`.
5. Reject duplicate `id`.
6. Convert labels to integer IDs through fixed mappings.

Training image lookup:

1. Join `image_dir / image_id`.
2. Confirm file exists.
3. Load image as RGB.
4. Ensure or transform to 64x64 according to implementation setting.

#### Dependencies

- Python standard library CSV/path utilities.
- Image library selected by implementation, likely PIL through Pillow.
- PyTorch dataset utilities if PyTorch is used.

#### Failure Handling

- Missing CSV: fail with path.
- Missing required column: fail with column name.
- Duplicate ID: fail with duplicated filename.
- Unknown label: fail with row ID and bad label.
- Missing image: fail with image path.
- Bad image: fail with image path and decoder error.

#### Independent Test Plan

- Tiny valid train CSV fixture parses.
- Tiny valid generate CSV fixture parses.
- Missing required column fails.
- Duplicate ID fails.
- Unknown label fails.
- Missing image fails.
- Valid RGB 64x64 PNG loads.
- Stable label mapping across repeated calls.

#### Open Questions

- Whether training images should be strictly required to already be 64x64 or resized to 64x64 during loading.

### Model and Diffusion Module

#### Responsibility

Own the from-scratch conditional denoising model, timestep/condition embedding contract, diffusion schedule, training loss, sampling procedure, and optional EMA weight handling.

#### Non-Responsibility

- No CSV parsing.
- No CLI argument parsing.
- No filesystem output naming.
- No README or package documentation.
- No local scorer execution.

#### Inputs and Outputs

Training inputs:

- noisy image tensor.
- clean image tensor.
- timestep tensor.
- animal ID tensor.
- object ID tensor.

Training outputs:

- predicted noise tensor.
- scalar loss from diffusion objective.

Sampling inputs:

- animal ID tensor.
- object ID tensor.
- sample count.
- sampling settings.

Sampling outputs:

- generated image tensor convertible to RGB PNG at 64x64.

#### Public Interface

Design-level interface:

- `build_model(model_config, label_maps) -> model`
- `build_diffusion(diffusion_config) -> diffusion`
- `diffusion.training_loss(model, clean_images, animal_ids, object_ids) -> loss`
- `diffusion.sample(model, animal_ids, object_ids, sampling_config) -> image_tensor`
- `create_ema(model, decay) -> ema_state` if EMA is implemented.

#### Data Structures

- `ModelConfig`: image size, channels, condition embedding size, timestep embedding size, model capacity settings.
- `DiffusionConfig`: number of timesteps, beta/noise schedule name and parameters.
- `SamplingConfig`: number of sampling steps, seed/random generator behavior, EMA usage.
- `EMAState`: moving average weights and decay if enabled.

#### Internal Design

- Use a compact UNet-style denoiser trained from scratch.
- Embed timestep, animal ID, and object ID.
- Combine condition embeddings and inject them into residual blocks.
- Predict noise for MSE training.
- Use direct pixel-space 64x64 RGB diffusion.
- Keep optional EMA as an enhancement after baseline works.

#### Algorithm Details

Training loss:

1. Sample timestep `t`.
2. Sample Gaussian noise.
3. Produce noisy image using the forward diffusion schedule.
4. Predict noise from noisy image, timestep, animal ID, and object ID.
5. Compute MSE between predicted noise and sampled noise.

Sampling:

1. Start from Gaussian noise.
2. Iterate reverse diffusion steps using trained model predictions.
3. Clamp or convert final tensor to valid image range.
4. Return generated tensor for the Generation CLI to save.

Exact beta schedule, channel counts, block counts, attention placement, and sampling step count remain implementation choices to tune and document.

#### Dependencies

- PyTorch, assuming the approved implementation stack remains Python/PyTorch.
- No pretrained generator dependencies.
- No `diffusers` pipeline or high-level generation training flow.

#### Failure Handling

- Invalid label tensor shape: fail before model call when possible.
- Invalid timestep range: fail or assert during development.
- Non-finite loss: training loop should stop or report failure.
- Checkpoint/config mismatch: generation must fail before sampling.

#### Independent Test Plan

- Forward pass returns tensor shape matching input image shape.
- Training loss is scalar and finite for tiny random batch.
- Diffusion schedule tensors have expected length and finite values.
- Cumulative alpha behavior is monotonic in the expected direction.
- Sampler with tiny step count returns correct batch/image shape.
- Backpropagation reaches trainable parameters.

#### Open Questions

- Exact UNet capacity.
- Exact diffusion beta schedule.
- Exact training timesteps and sampling steps.
- Whether EMA is in the first implementation or added after baseline validation.

### Training CLI

#### Responsibility

Provide the user-facing training command, construct data/model/diffusion components, run training, optionally run a small overfit workflow, save checkpoints, and record reproducibility metadata.

#### Non-Responsibility

- No generated image contract validation.
- No Codabench packaging.
- No local scorer execution.
- No final choice of leaderboard-tuning strategy.

#### Inputs and Outputs

Inputs:

- training CSV path.
- training image directory.
- checkpoint output path.
- model/diffusion/training settings.
- seed.
- optional small-overfit mode settings.

Outputs:

- checkpoint file.
- optional sample previews for manual inspection.
- training log or console output.

#### Public Interface

Proposed command:

```bash
python scripts/train.py --train_csv dataset/train.csv --image_dir dataset/trainset --checkpoint model.pth --seed 0
```

Optional command capabilities:

- `--data_check_only`
- `--overfit_subset N`
- `--epochs` or `--steps`
- `--batch_size`
- `--lr`
- `--device`
- `--sample_dir` for previews

Exact argument names can be adjusted during implementation, but README must match final names.

#### Data Structures

- `TrainingConfig`: paths, seed, batch size, learning rate, training length, device, checkpoint path, model/diffusion config.
- `RunMetadata`: timestamp or run ID if used, seed, command-relevant settings, package versions if captured.
- checkpoint payload following the shared checkpoint contract.

#### Internal Design

- Parse CLI arguments.
- Set random seeds where supported.
- Load and validate training rows.
- Build dataset and dataloader.
- Build model and diffusion.
- Run training loop.
- Optionally update EMA.
- Save checkpoint with all generation-critical settings.

#### Algorithm Details

Training loop:

1. For each batch, move images and labels to device.
2. Compute diffusion training loss.
3. Backpropagate.
4. Step optimizer.
5. Optionally update EMA.
6. Periodically log loss and optionally save preview samples.
7. Save final checkpoint.

Small overfit mode:

1. Restrict dataset to a tiny deterministic subset.
2. Train enough iterations to show the loss path and sampling mechanics work.
3. Save a test checkpoint or previews separate from final checkpoint.

#### Dependencies

- Data and Label Module.
- Model and Diffusion Module.
- PyTorch optimizer/dataloader utilities.

#### Failure Handling

- Missing data path: fail before training.
- Invalid CSV/image data: fail before model construction.
- Device unavailable: fail clearly or fall back only if explicitly designed.
- Non-finite loss: stop and report training failure.
- Checkpoint write failure: fail with output path.

#### Independent Test Plan

- CLI argument parser accepts minimal required paths.
- `--data_check_only` or equivalent validates real/fixture data without training.
- Tiny fixture dry run performs one optimization step.
- Checkpoint save includes required keys.
- Missing required paths fail with clear messages.

#### Open Questions

- Final training command and supported arguments.
- Exact test framework used for CLI tests.
- Whether CPU one-step smoke tests are required even if final training needs GPU.

### Generation CLI

#### Responsibility

Provide the user-facing generation command. Load a checkpoint, load `dataset/generate.csv`, sample one image per row, and save exactly the requested filenames into the output directory.

#### Non-Responsibility

- No model training.
- No output-contract validation beyond generation-time sanity checks.
- No local scoring.
- No upload/package creation.

#### Inputs and Outputs

Inputs:

- checkpoint path.
- generation CSV path.
- output directory.
- seed or generation randomness policy.
- optional device and sampling settings.

Outputs:

- generated PNG files at `output_dir/{id}`.

#### Public Interface

Proposed command:

```bash
python scripts/generate.py --checkpoint model.pth --generate_csv dataset/generate.csv --output_dir generated_images --seed 0
```

Optional command capabilities:

- `--device`
- `--batch_size`
- `--sampling_steps`
- `--use_ema`
- `--overwrite` or explicit stale-output policy

#### Data Structures

- `GenerationConfig`: checkpoint path, generation CSV, output directory, seed, batch size, device, sampling settings.
- `GenerateRow`: from Data and Label Module.
- generated image tensor batches from Model and Diffusion Module.

#### Internal Design

- Parse CLI arguments.
- Load checkpoint and validate required metadata.
- Load label maps from checkpoint.
- Load and validate generation rows.
- Build model/diffusion from checkpoint metadata.
- Sample images in batches grouped by generation-row order.
- Save each output as RGB PNG using exact `id` filename.
- Avoid silently accepting stale extra files; either require an empty/nonexistent output directory or document an explicit overwrite policy.

#### Algorithm Details

Generation loop:

1. Read all generation rows.
2. Convert labels to IDs using checkpoint-compatible mappings.
3. For each batch of rows:
   - sample image tensors.
   - convert tensors to RGB image values.
   - save each image as `output_dir / image_id`.
4. Report generated count.

#### Dependencies

- Data and Label Module.
- Model and Diffusion Module.
- checkpoint contract from Training CLI.
- image saving library.

#### Failure Handling

- Missing checkpoint: fail before generation.
- Invalid checkpoint metadata: fail with missing key/config.
- Unknown generation label: fail before sampling.
- Duplicate generation ID: fail before sampling.
- Output path collision/stale files: fail or require explicit overwrite policy.
- Image save failure: fail with filename.

#### Independent Test Plan

- Missing checkpoint fails.
- Incompatible checkpoint mapping fails.
- Tiny valid generation fixture writes exact filenames.
- Duplicate generation ID fails.
- Generation preserves exact filename strings.
- Fixture output can be passed into Generated Output Validator.

#### Open Questions

- Whether generation output directory must be empty or can be overwritten with an explicit flag.
- Exact deterministic behavior guarantee for sampling with a fixed seed across hardware.

### Generated Output Validator

#### Responsibility

Provide a fast project-owned quality gate that verifies generated image files satisfy the assignment output contract.

#### Non-Responsibility

- No model quality judgment.
- No FID/CLIP-T scoring.
- No image generation.
- No training.

#### Inputs and Outputs

Inputs:

- generation CSV path.
- generated image directory.
- expected image size, default 64.
- expected count, default 2,000 for assignment mode.

Outputs:

- pass/fail exit status.
- human-readable error report.

#### Public Interface

Proposed command:

```bash
python scripts/validate_generated.py --generate_csv dataset/generate.csv --image_dir generated_images --image_size 64 --expected_count 2000
```

Fixture tests may use a lower `--expected_count`.

#### Data Structures

- expected filename set from `generate.csv`.
- actual filename set from output directory.
- validation error list.

#### Internal Design

- Read expected filenames from generation CSV.
- Reject duplicate expected IDs.
- List files in generated image directory.
- Compare exact sets for missing and extra files.
- Open each expected image.
- Verify format, mode, and size.
- Print all detected errors when practical, then exit nonzero on failure.

#### Algorithm Details

Validation:

1. Load expected IDs.
2. Confirm expected count.
3. List actual files.
4. Compute missing and extra sets.
5. For each expected file:
   - open image.
   - verify PNG.
   - verify RGB.
   - verify size `(64, 64)`.
6. Pass only if all checks pass.

#### Dependencies

- Python standard library CSV/path utilities.
- PIL/Pillow or equivalent image reader.

#### Failure Handling

- Missing CSV or image directory: fail.
- Duplicate IDs: fail.
- Missing files: fail and list examples.
- Extra files: fail and list examples.
- Bad image decode: fail with filename.
- Wrong mode/format/size: fail with filename and observed value.

#### Independent Test Plan

- Valid tiny fixture passes.
- Missing file fails.
- Extra file fails.
- Duplicate generation ID fails.
- Grayscale PNG fails.
- RGBA PNG fails.
- RGB 32x32 PNG fails.
- Non-PNG file with expected name fails.

#### Open Questions

- Whether `.png` filename extension case should be accepted exactly as CSV or normalized. Default design: exact CSV filename match.

### Documentation and Environment Manifest

#### Responsibility

Make the final project reproducible by documenting dependencies, training, generation, validation, checkpoint use, and any extra data or auxiliary pretrained modules.

#### Non-Responsibility

- No model training.
- No validation execution.
- No package upload.

#### Inputs and Outputs

Inputs:

- final script paths and arguments.
- final dependency versions.
- final checkpoint name/path.
- final training and generation settings.
- local scoring notes if used.

Outputs:

- `README.md`
- `requirements.txt` or equivalent approved environment manifest.

#### Public Interface

No runtime API. The required public contract is document content:

- setup command.
- training command.
- generation command.
- validation command.
- checkpoint used.
- package versions.
- seed/config.
- extra data/pretrained auxiliary note.

#### Data Structures

- README sections.
- dependency manifest lines.

#### Internal Design

- Keep README focused on reproduction.
- Avoid storing secrets, private paths, or credential values.
- Mention generated images/checkpoints as artifacts, not source files.
- Keep local scorer caveats clear if scorer setup remains unresolved.

#### Algorithm Details

Not applicable.

#### Dependencies

- Final implementation choices.
- Training CLI, Generation CLI, Generated Output Validator.

#### Failure Handling

- Documentation checklist fails if any required command or checkpoint detail is missing.
- Requirements manifest remains incomplete until versions are chosen.

#### Independent Test Plan

- Review checklist verifies README contains setup, train, generate, validate, checkpoint, seed/config, and extra-data note.
- Command consistency check compares documented script paths to implemented files.
- Git status review confirms generated artifacts are not accidentally staged.

#### Open Questions

- Exact Python/CUDA/PyTorch versions.
- Exact dependency pins.
- Final checkpoint filename.

### Optional Local Scoring Adapter

#### Responsibility

Document or prepare the local scorer workflow once scorer inputs are complete. Keep local scoring separate from core generation and validation.

#### Non-Responsibility

- No implementation of FID or CLIP-T.
- No changes to official scorer behavior.
- No claim that local scoring is verified before it is run.
- No generation output validation replacement.

#### Inputs and Outputs

Inputs:

- validated generated images.
- `scoring_program/score.py`.
- local scorer input layout.
- reference files including `test.json` and reference images when available.

Outputs:

- scorer-compatible input directory if implemented.
- local `scores.json` from scorer after an approved run.
- README notes.

#### Public Interface

Known command, not verified:

```bash
cd scoring_program
python score.py --input_dir ./input --output_dir ./ --image_size 64 --num_images 3000 --test_json test.json --score fid clip_t clip_i --verbose
```

This command must remain labeled not verified until actually run.

#### Data Structures

- scorer input layout:
  - `scoring_program/input/ref`
  - `scoring_program/input/res`
- scorer output:
  - `scores.json`

#### Internal Design

- First check whether required local scorer inputs exist.
- Do not run scorer without user approval.
- Do not treat 3,000-image local scorer output as submission-equivalent until the mismatch with 2,000 generated images is resolved.
- Keep scorer outputs out of git.

#### Algorithm Details

Not applicable; official scorer owns scoring algorithms.

#### Dependencies

- Provided scoring program.
- Generated Output Validator pass result.
- Scorer dependencies such as `torch`, `open_clip`, `torchvision`, `numpy`, `scipy`, `PIL`, and `tqdm`.

#### Failure Handling

- Missing `test.json` or reference images: do not run; report setup incomplete.
- Missing generated images: do not run.
- Missing dependencies or GPU: report environment issue.
- 2,000-vs-3,000 mismatch: report diagnostic limitation.

#### Independent Test Plan

- Setup check detects missing reference files.
- Documentation check labels scorer command as known/not run until verified.
- Git status review excludes `scores.json`.

#### Open Questions

- How to complete local scorer reference setup.
- Whether development scoring should use a 2,000-image adaptation or the discovered 3,000-image command.

## Cross-Module Contracts

### Data to Training

The Data and Label Module returns normalized image tensors and integer label IDs. Training CLI owns batching and optimization. The model never parses CSV files directly.

### Training to Checkpoint

Training CLI must save all metadata required by Generation CLI. Generation must not depend on hardcoded training defaults that are missing from the checkpoint or README.

### Checkpoint to Generation

Generation CLI rebuilds model and diffusion objects from checkpoint metadata. It must reject missing or incompatible metadata before sampling.

### Generation to Validator

Generation CLI writes files. Generated Output Validator independently checks the output directory. Passing generation logs is not enough; validator pass is required.

### Validator to Local Scoring

Local scoring should only run after validator pass. Scoring cannot replace validator because the validator checks assignment packaging details cheaply and deterministically.

## End-to-End Workflow

### Development Data Check

1. Run a data-check path in Training CLI or a small dedicated check.
2. Confirm `train.csv` row count, `generate.csv` row count, label sets, and training image existence.

### Tiny Overfit Check

1. Train on a tiny deterministic subset.
2. Confirm loss is finite and can decrease enough to demonstrate wiring.
3. Generate sample previews for manual inspection.
4. Save a non-final checkpoint.

### Full Training

1. Train on all 4,799 rows.
2. Save final checkpoint with metadata.
3. Record command and environment in README.

### Final Generation

1. Run Generation CLI with final checkpoint and `dataset/generate.csv`.
2. Write outputs to `generated_images/`.
3. Run Generated Output Validator.

### Optional Local Scoring

1. Confirm local scorer inputs are complete.
2. Arrange generated images as required by scorer.
3. Run scorer only with approval.
4. Record scores as diagnostic, not as verified official grade unless setup is known equivalent.

## Test Strategy Mapping

| Test Plan Requirement | Coverage Location |
| --- | --- |
| Repository data presence | Data and Label Module tests; Training CLI data-check path |
| `train.csv` row count 4,799 | Data and Label Module validation |
| `generate.csv` row count 2,000 | Data and Label Module and Generated Output Validator |
| Class coverage for 10 animals/10 objects | Data and Label Module label tests |
| Training CLI startup | Training CLI parser/smoke test |
| Generation CLI startup | Generation CLI parser/smoke test |
| Validator startup | Generated Output Validator fixture test |
| CSV schema tests | Data and Label Module unit tests |
| Image loading tests | Data and Label Module unit tests |
| Model tensor-shape tests | Model and Diffusion Module unit tests |
| Diffusion schedule sanity | Model and Diffusion Module unit tests |
| Loss finite/backpropagates | Model and Diffusion Module unit tests |
| Sampler shape/range | Model and Diffusion Module unit tests |
| Checkpoint contract | Training CLI and Generation CLI integration tests |
| Output naming | Generation CLI tests and Generated Output Validator |
| Missing/extra/wrong-mode/wrong-size images | Generated Output Validator tests |
| README reproducibility checklist | Documentation and Environment Manifest review |
| Local scorer setup status | Optional Local Scoring Adapter checks |
| Data-to-training dry run | Integration test |
| Training-to-generation compatibility | Integration test |
| Generation-to-validation pipeline | Integration test |
| Full dataset contract check | Data and Label Module integration/data-check path |
| Golden test cases | Data, Generation, and Validator fixture tests |
| Randomized/property tests | Data mapping, validator set equality, sampler shape tests |
| Performance benchmarks | Training CLI and optional scorer, thresholds open |
| Regression tests | Validator, generation, checkpoint, README tests after implementation lands |
| Manual verification | Sample grid inspection, README/provenance/git review |

## Quality Gates

Known quality-gate status:

- No verified build/test/lint/format/type-check commands exist yet.
- The local scorer command is discovered but not verified.
- The project-owned generated-output validator is missing and must be implemented.

Minimum implementation gates:

1. Data check passes for assignment CSVs and training images.
2. Unit checks pass for CSV parsing, labels, model shape/loss, checkpoint metadata, and validator cases.
3. Tiny integration run saves a checkpoint and generates fixture outputs.
4. Generated Output Validator passes for final `generated_images/`.
5. README includes environment, training, generation, validation, checkpoint, seed/config, and extra-data note.
6. Git status review confirms generated artifacts and scorer outputs are not staged unless packaging is explicitly requested.

Known evaluator command, not run:

```bash
cd scoring_program
python score.py --input_dir ./input --output_dir ./ --image_size 64 --num_images 3000 --test_json test.json --score fid clip_t clip_i --verbose
```

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Invalid generated file contract wastes Codabench upload | Make Generated Output Validator mandatory before upload |
| Checkpoint lacks metadata needed for generation | Enforce checkpoint contract in Training and Generation CLI tests |
| Model trains but ignores labels | Inspect per-pair sample grids; keep animal/object conditioning explicit |
| Local scoring setup is misleading | Keep scorer optional and label command known/not verified until run |
| Pixel-space DDPM sampling is slow | Keep sampling steps configurable and document final settings |
| Compact model underperforms | Start with baseline; tune capacity only after validator and overfit checks pass |
| Extra data creates reproducibility burden | Default to no extra data; document provenance if this changes |
| Over-designed framework delays baseline | Keep scripts and modules minimal |

## Assumptions

- Implementation will use Python and PyTorch.
- First version uses no extra data.
- First version uses no pretrained auxiliary model in the main generation path.
- `generated_images/` is the canonical final output directory.
- `model.pth` or a README-documented checkpoint name is acceptable.
- Fixed assignment label order is acceptable for deterministic label mapping.
- Test fixtures may use smaller row counts than the full assignment data.
- Exact script/function names may change during implementation if README and tests are updated consistently.

## Open Questions

- What Python, CUDA, PyTorch, and package-manager versions should be used?
- Should the implementation use `pytest`, `unittest`, or no-framework assertion scripts?
- Should training image loading strictly require 64x64 inputs or resize to 64x64?
- What GPU/runtime budget should define practical performance thresholds?
- Which exact UNet size, beta schedule, training steps, and sampling steps should be the first training run?
- Should EMA be implemented in the first pass or added after baseline validation?
- Should generation require an empty output directory or support explicit overwrite?
- How deterministic must generation be across different GPU hardware for the same seed?
- How should `hw6_reference.zip` be unpacked or arranged to complete local scoring?
- Should local development scoring adapt to 2,000 images or keep the discovered 3,000-image scorer setup?
- Will extra data be used?
