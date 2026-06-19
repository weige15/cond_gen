# Implementation Prompt

## Objective

Implement the HW6 Brainrot conditional image generation project end to end.

Build a small, reproducible Python/PyTorch pipeline that trains a from-scratch conditional image generator on `dataset/train.csv` and `dataset/trainset/`, then generates exactly 2,000 RGB PNG images at 64x64 using `dataset/generate.csv`. Generated filenames must exactly match the `id` values in `dataset/generate.csv`.

Default architecture direction: a minimal label-conditioned DDPM with a from-scratch UNet-style denoiser, explicit animal/object conditioning, a project-owned training loop, a generation CLI, and a generated-output validator.

## Inputs to Read First

Read these before changing code:

- `AGENTS.md` - repository guardrails, assignment constraints, command permission rules, and git/artifact rules.
- `doc/problem-brief.md` - present; assignment facts, dataset counts, official constraints, scoring bands, and open questions.
- `doc/repo-map.md` - present; current repository layout and missing implementation.
- `doc/quality-gates.md` - present; discovered scorer command and missing build/test/lint/type gates.
- `doc/proposal.md` - required; selected baseline approach and validation plan.
- `doc/high-level-design.md` - required; module boundaries and contracts.
- `doc/test-plan.md` - required; planned unit, smoke, integration, and validator checks.
- `doc/detailed-design.md` - required; implementation-level data, checkpoint, CLI, model, and validator contracts.
- `doc/tasks/` - required; module task files:
  - `doc/tasks/data-and-label-module.md`
  - `doc/tasks/model-and-diffusion-module.md`
  - `doc/tasks/training-cli.md`
  - `doc/tasks/generation-cli.md`
  - `doc/tasks/generated-output-validator.md`
  - `doc/tasks/documentation-and-environment-manifest.md`
  - `doc/tasks/optional-local-scoring-adapter.md`
  - `doc/tasks/progress.md`
- `conditional_score_diffusion_heuristics.md` - supplemental implementation notes only; do not copy repository code from external sources without license/provenance handling.
- `README.md` - currently minimal and must be replaced with reproducibility instructions during implementation.
- `.gitignore` - currently ignores `dataset/`, `scoring_manual.txt`, `*.zip`, `*:Zone.Identifier`, and `scoring_program/`.
- `scoring_manual.txt` and `scoring_program/score.py` - read only if working on scoring documentation/setup.

## Current Implementation

Current repository root: `/home/kuotzuwei15/GenAI/hw6`.

Git state observed during prompt generation:

- Branch: `main`.
- Remote: `origin git@github.com:weige15/cond_gen.git`.
- `doc/` is untracked.
- Tracked source-like files are limited to `.gitignore`, `AGENTS.md`, `HW6-Brainrot Image Generation.pdf`, `README.md`, and `conditional_score_diffusion_heuristics.md`.

Current codebase facts:

- No project implementation exists yet.
- No `scripts/` directory exists.
- No model, dataset loader, training script, generation script, validator script, package module, tests, checkpoint, or generated final output exists.
- No `pyproject.toml`, `setup.py`, `requirements.txt`, lockfile, `Makefile`, CI config, lint config, type-check config, or test config exists.
- `README.md` currently contains only `# cond_gen`.
- `dataset/train.csv` exists with columns `id,animal,object`; local planning docs record 4,799 data rows.
- `dataset/generate.csv` exists with columns `id,animal,object,prompt`; local planning docs record 2,000 data rows.
- `dataset/trainset/` exists and contains 4,799 PNG training images.
- `scoring_program/score.py` exists as the provided local evaluator, not a project test suite.
- `scoring_program/input/ref/config.json` sets `image_size: 64`, `num_images: 3000`, and scores `fid`, `clip_t`, `clip_i`.
- `scoring_program/input/ref/test_mu.npy` and `test_sigma.npy` exist.
- `scoring_program/input/ref/test.json` and `scoring_program/input/ref/test/` were not observed, so local CLIP/CLIP-I scoring setup is incomplete.
- `scoring_program/input/res/` exists but is empty.

Discovered evaluator command, not verified:

```bash
cd scoring_program
python score.py --input_dir ./input --output_dir ./ --image_size 64 --num_images 3000 --test_json test.json --score fid clip_t clip_i --verbose
```

Do not treat the scorer as a normal test gate until setup is complete, generated images exist, and the user approves running it. It writes `scores.json`, may require GPU, and may load/download model weights.

## Hard Constraints

- Main generator must be trained from scratch.
- Do not use pretrained generator weights, pretrained UNet/Transformer/diffusion/generation weights, or high-level pretrained generation pipelines for the main model.
- Do not use `diffusers` pipelines or high-level ready-made generation training flows for the main model.
- Pretrained CLIP or VAE may be used only as auxiliary modules, not to generate final images or replace the main generator.
- Keep animal ID and object ID conditioning explicit and derived from CSV labels.
- Final generated output must be exactly 2,000 PNG files.
- Every generated image must be RGB and exactly 64x64.
- Output filenames must exactly match `dataset/generate.csv`.
- Keep generated images, checkpoints, datasets, scorer outputs, archives, and large training artifacts out of source changes unless the user explicitly asks for packaging.
- Do not overwrite dataset, PDF, zips, scoring references, or generated results without explicit permission.
- Do not push to `git@github.com:weige15/cond_gen.git` without explicit user permission.
- Follow `AGENTS.md` for commands requiring permission, especially training, generation, scoring, installing dependencies, large output creation, git operations, and destructive actions.
- Preserve assignment compliance and reproducibility in `README.md`.

## Non-Goals

- Do not build a general research framework.
- Do not support multiple model families in the first implementation.
- Do not automate Codabench upload.
- Do not rewrite the official scorer.
- Do not make local scoring final-submission-equivalent until missing reference inputs and the 2,000-vs-3,000 mismatch are resolved.
- Do not add extra data unless the user explicitly chooses it and provenance is documented.
- Do not refactor unrelated files.

## Execution Model

1. Read all required docs listed above before editing.
2. Maintain `doc/tasks/progress.md` throughout implementation. Add timestamped or checkpoint-style entries for started, completed, blocked, and verified work after each small workstream.
3. Implement one module or small workstream at a time.
4. Prefer the smallest HW6-specific Python/PyTorch implementation that satisfies the contracts.
5. Use subagents only for independent workstreams with disjoint write scopes and low integration cost.
6. Keep write scopes disjoint when using subagents. Shared files such as `README.md`, dependency manifests, and common modules should be integrated by the main agent.
7. Do not revert edits made by the user or other agents. Inspect existing changes before modifying touched files.
8. Run module-specific checks after each module. If no test command exists yet, add the smallest useful check for the module.
9. Run full available quality gates at the end.
10. Summarize command outputs as evidence in progress notes and final response.
11. Stop and ask only when blocked by missing requirements, destructive choices, credentials, external services, permission-sensitive commands, or contradictory docs.

## Module Workstreams

### Data and Label Module

Expected files:

- Shared helper module under `scripts/` or a small package directory chosen by the implementer.
- Focused data checks or tests.

Ownership:

- CSV parsing for `dataset/train.csv` and `dataset/generate.csv`.
- Fixed animal label order: `shark`, `crocodile`, `frog`, `cat`, `dog`, `capybara`, `elephant`, `bird`, `fish`, `monkey`.
- Fixed object label order: `sneaker`, `airplane`, `coffee cup`, `banana`, `cactus`, `toilet`, `pizza`, `drum`, `car`, `chair`.
- Deterministic label maps.
- Training image lookup/loading.
- Full-data validation for 4,799 train rows, 2,000 generation rows, labels, duplicates, and referenced image files.

Checks:

- Valid tiny CSV fixtures parse.
- Missing columns, duplicate IDs, unknown labels, empty fields, and missing images fail clearly.
- Full assignment data check passes before training.

### Model and Diffusion Module

Expected files:

- Shared model/diffusion module, likely `scripts/model.py` or equivalent.
- Focused tensor checks or tests.

Ownership:

- From-scratch conditional denoiser.
- Timestep, animal, and object conditioning.
- Diffusion schedule.
- MSE noise-prediction training loss.
- Reverse sampler.
- Optional EMA only after baseline wiring works.

Checks:

- Forward output shape matches input image tensor shape.
- Loss is scalar, finite, and backpropagates.
- Schedule tensors are finite and monotonic where expected.
- Tiny-step sampler returns RGB-shaped 64x64 tensors.
- No pretrained generator weights or high-level generation framework is introduced.

### Training CLI

Expected files:

- `scripts/train.py`
- Shared checkpoint helper if needed.
- Training smoke/integration checks.

Proposed command shape:

```bash
python scripts/train.py --train_csv dataset/train.csv --image_dir dataset/trainset --checkpoint model.pth --seed 0
```

Ownership:

- CLI argument parsing.
- Seed/device handling.
- Data-check-only mode.
- Dataset/model/diffusion/optimizer wiring.
- Training loop.
- Small overfit mode.
- Checkpoint save with model weights, configs, label maps, seed, training settings, and optional EMA metadata.

Checks:

- CLI starts and rejects missing paths clearly.
- Data-check-only path validates real assignment data without training.
- Tiny fixture dry run performs one optimization step.
- Saved checkpoint contains required generation metadata.

### Generation CLI

Expected files:

- `scripts/generate.py`
- Generation integration checks.

Proposed command shape:

```bash
python scripts/generate.py --checkpoint model.pth --generate_csv dataset/generate.csv --output_dir generated_images --seed 0
```

Ownership:

- Load and validate checkpoint metadata.
- Load `dataset/generate.csv`.
- Reject duplicate IDs or checkpoint-incompatible labels.
- Sample images in batches.
- Save exact `output_dir/{id}` RGB PNG filenames.
- Define stale-output behavior: require an empty/nonexistent output directory or an explicit overwrite flag.

Checks:

- Missing/incompatible checkpoint fails clearly.
- Tiny generation fixture writes exact requested filenames.
- Fixture generated output passes validator.

### Generated Output Validator

Expected file:

- `scripts/validate_generated.py`

Proposed command shape:

```bash
python scripts/validate_generated.py --generate_csv dataset/generate.csv --image_dir generated_images --image_size 64 --expected_count 2000
```

Ownership:

- Read expected filenames from generation CSV.
- Reject duplicate generation IDs.
- Compare exact expected and actual filename sets.
- Verify PNG format, RGB mode, and 64x64 size.
- Print actionable error examples and exit nonzero on any failure.

Checks:

- Valid tiny fixture passes.
- Missing file, extra file, duplicate ID, grayscale/RGBA image, wrong size, and non-PNG data fail.
- Final `generated_images/` must pass before scoring or upload.

### Documentation and Environment Manifest

Expected files:

- `README.md`
- `requirements.txt` or another explicit environment manifest if chosen.

Ownership:

- Environment setup.
- Exact training, generation, and validation commands.
- Checkpoint name/path used for final generation.
- Seed/config.
- Package versions.
- Assignment-compliance notes: no pretrained generator, no extra data unless documented, local scorer caveats.
- Artifact policy: generated images and checkpoints stay out of git unless packaging is explicitly requested.

Checks:

- README command paths match implemented scripts.
- README includes setup, train, generate, validate, checkpoint, seed/config, dependencies, extra-data/pretrained-module notes, and scoring caveats.
- Git status review confirms generated artifacts are not staged accidentally.

### Optional Local Scoring Adapter

Expected files:

- README scoring notes.
- Optional lightweight setup-check script only if it is genuinely useful.

Ownership:

- Keep the known local scorer command documented as known/not run until verified.
- Check/report required scorer inputs: `input/ref/test.json`, `input/ref/test/`, `test_mu.npy`, `test_sigma.npy`, generated results in `input/res`, dependencies, and GPU expectations.
- Keep the 2,000-vs-3,000 mismatch explicit.
- Run scorer only after validator pass and explicit user approval.

Checks:

- Do not claim evaluator verification before a real run.
- Missing scorer reference inputs are reported as setup incomplete.
- Scorer outputs such as `scores.json` remain out of source changes unless requested.

## Subagent Plan

Subagents are useful only if their write scopes are kept separate.

Good subagent candidates:

- Data and Label Module: may edit the shared data helper and data-specific checks only.
- Model and Diffusion Module: may edit model/diffusion helper and tensor checks only.
- Generated Output Validator: may edit `scripts/validate_generated.py` and validator fixtures/checks only.
- Documentation and Environment Manifest: may draft README sections after script interfaces are stable.

Keep in the main agent:

- Final file layout decisions.
- Shared dependency manifest.
- Checkpoint contract integration between training and generation.
- `scripts/train.py` and `scripts/generate.py` integration.
- `doc/tasks/progress.md` updates.
- Final quality-gate run and git status review.

Shared or integration-only files:

- `README.md`
- `requirements.txt` or equivalent manifest
- Common helper modules used by multiple CLIs
- `doc/tasks/progress.md`

Merge approach:

- Let each subagent work on one bounded module.
- Main agent reads each result, checks it against `doc/detailed-design.md`, runs that module's checks, and then integrates shared imports/interfaces.
- Do not allow two agents to write the same file concurrently.

## Implementation Order

1. Repository setup and progress tracking.
   - Read docs and current files.
   - Update `doc/tasks/progress.md` with the implementation start checkpoint.
   - Decide the minimal Python file layout.
   - Verification: `git status --short --branch` and a progress entry.

2. Data and Label Module.
   - Implement fixed label maps, CSV loaders, validation, and image loading.
   - Add tiny fixture checks and a full assignment data-check path.
   - Verification: run the module-specific data checks. If the command is newly created, record it in progress and README draft notes.

3. Generated Output Validator.
   - Implement `scripts/validate_generated.py`.
   - Add fixture checks for valid, missing, extra, duplicate, wrong mode, wrong size, and non-PNG cases.
   - Verification: run validator fixture checks.

4. Model and Diffusion Module.
   - Implement compact from-scratch conditional denoiser, diffusion loss, schedule, and tiny-step sampler.
   - Verification: run tensor/loss/sampler checks on CPU or the available device.

5. Training CLI.
   - Implement `scripts/train.py`, data-check-only mode, tiny dry run, small overfit option, full training path, and checkpoint contract.
   - Verification: run CLI startup/missing-path checks, data-check-only, and tiny one-step checkpoint check.

6. Generation CLI.
   - Implement `scripts/generate.py`, checkpoint loading, generation CSV loading, stale-output policy, batch sampling, exact filename saving.
   - Verification: run tiny checkpoint-to-generation fixture and pass validator fixture mode.

7. Documentation and environment manifest.
   - Update `README.md`.
   - Add `requirements.txt` or chosen environment manifest after versions are known.
   - Verification: documentation checklist and command consistency review.

8. Full local workflow gates.
   - Run full assignment data check.
   - Run training only with user permission because it may consume significant GPU time and create checkpoints.
   - Run final 2,000-image generation only with user permission because it creates `generated_images/`.
   - Run `scripts/validate_generated.py` on final output.
   - Run local scorer only with user permission and only if scorer inputs are complete.

## Testing and Quality Gates

Existing configured gates:

- Build command: missing.
- Unit test command: missing.
- Integration test command: missing.
- Lint command: missing.
- Format command: missing.
- Type-check command: missing.
- Static-analysis command: missing.
- Project smoke-test command: missing.
- Evaluator command: discovered but not verified.

Minimum gates to add or run during implementation:

- Data check for full assignment inputs: 4,799 train rows, 2,000 generation rows, valid labels, duplicate detection, and referenced training image existence.
- Validator fixture checks: exact filename set, duplicate ID, missing/extra file, PNG/RGB/64x64 failures.
- Model checks: forward shape, finite scalar loss, backpropagation, schedule sanity, tiny-step sampler shape.
- Training CLI smoke: parser startup, missing paths, data-check-only, one-step tiny dry run, checkpoint required keys.
- Generation CLI smoke: missing checkpoint, incompatible checkpoint metadata, tiny generation fixture, validator pass.
- README checklist: setup, training, generation, validation, checkpoint, seed/config, dependency versions, no-extra-data/pretrained-generator note, scoring caveat.
- Git status review: no datasets, generated images, checkpoints, scorer outputs, archives, or unrelated files staged accidentally.

If a test framework is introduced, document the command in README and `doc/tasks/progress.md`. If no framework is introduced, use the smallest runnable assertion/check scripts that fail on the contracts above.

Known evaluator command, optional and not verified:

```bash
cd scoring_program
python score.py --input_dir ./input --output_dir ./ --image_size 64 --num_images 3000 --test_json test.json --score fid clip_t clip_i --verbose
```

Run this only after:

- `generated_images/` or scorer `input/res` has valid images.
- `scoring_program/input/ref/test.json` and `scoring_program/input/ref/test/` are present.
- Dependencies/GPU are available.
- The 2,000-vs-3,000 mismatch is understood.
- The user approves the run.

## Progress Tracking

Maintain `doc/tasks/progress.md`.

For each module or small workstream, record:

- Start checkpoint with date/time or clear sequence label.
- Files planned for editing.
- Completion status.
- Checks run, with command summaries and pass/fail results.
- Blockers and assumptions.
- Verification status.

Update progress after each module, not only at the end. If a command is skipped, record why.

## Commit or Checkpoint Strategy

Do not commit unless the user asks for commits.

If commits are requested:

- Inspect `git status --short --branch` first.
- Stage only related files.
- Keep commits logical and task-specific, for example `feat: add generated image validator`.
- Do not include datasets, generated images, checkpoints, scorer outputs, archives, or unrelated user changes.
- Do not push without explicit user permission.

If commits are not requested:

- Keep the final diff grouped by module.
- Preserve progress notes in `doc/tasks/progress.md`.
- End with a clear changed-file summary.

## Acceptance Criteria

Implementation is done when:

- Required behavior is implemented: train from scratch, generate from `dataset/generate.csv`, and validate exactly 2,000 RGB PNG 64x64 outputs with exact filenames.
- All module tasks in `doc/tasks/*.md` are completed or explicitly blocked with a documented reason.
- Tests/checks are added or updated for data parsing, label mapping, model/diffusion contracts, training checkpoint contract, generation output naming, validator failures, and README command consistency.
- Build, test, lint, format, type-check, and static-analysis gates pass when configured.
- If no such gates are configured, the implemented module smoke/check commands pass and are documented.
- Generated-output validator passes for final output before scoring or packaging.
- Evaluator/benchmark passes when configured and runnable; otherwise its limitations remain documented.
- README and dependency/environment manifest are updated for reproducibility.
- No unrelated files are changed.
- No datasets, generated images, checkpoints, scorer outputs, archives, or large artifacts are included in source changes unless explicitly requested.

## Uncertainty Protocol

Make conservative assumptions when safe and document them in `doc/tasks/progress.md` and README if they affect reproducibility.

Ask the user only when blocked by:

- Missing or conflicting assignment requirements.
- A destructive operation or overwrite risk.
- Installing dependencies.
- Running training, generation, scoring, or any command that creates large outputs or uses significant GPU time.
- Missing credentials or external services.
- Whether to use extra data.
- Whether to complete local scorer setup by unpacking/copying reference files.
- Quality gates that cannot be run and have no reasonable local substitute.

Known unresolved questions to preserve:

- Python/CUDA/PyTorch/package-manager versions are unspecified.
- GPU/runtime budget is unspecified.
- Whether to strictly require 64x64 training images or resize inputs during loading.
- Exact UNet capacity, beta schedule, training steps, sampling steps, and EMA timing are implementation/tuning choices.
- Local scorer has missing reference inputs and a 3,000-image setting while assignment generation requires 2,000 images.

## Final Response Requirements

At the end, provide a concise summary with:

- What was implemented.
- Changed files grouped by workstream.
- Tests and quality gates run, with command output summaries.
- Any commands not run and why.
- Known limitations or unresolved questions.
- Follow-up required from the user, especially for training duration, final generation, scorer setup, or Codabench packaging.

Do not include long logs. Do include enough command evidence for the user to know what passed.
