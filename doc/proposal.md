# Proposal: From-Scratch Conditional Brainrot Image Generation

## Objective

Build a reproducible HW6 solution that trains a conditional image generator from scratch and generates exactly 2,000 RGB PNG images at 64x64 from `dataset/generate.csv`.

The proposal target is a practical first implementation plan, not code. The default implementation direction is a small label-conditioned DDPM with a from-scratch UNet backbone, because it fits the assignment constraints and the local heuristic note while avoiding high-level pretrained generation pipelines.

## Source Inputs

- `HW6-Brainrot Image Generation.pdf`: primary assignment source for objective, dataset, restrictions, scoring, and submission format.
- `doc/problem-brief.md`: extracted assignment facts, dataset counts, scoring thresholds, and open questions.
- `doc/repo-map.md`: current repository state and missing implementation pieces.
- `doc/quality-gates.md`: discovered evaluator command and missing quality gates.
- `scoring_manual.txt`: local scoring setup and example command.
- `dataset/train.csv`: observed training metadata schema, `id,animal,object`.
- `dataset/generate.csv`: observed generation metadata schema, `id,animal,object,prompt`.
- `conditional_score_diffusion_heuristics.md`: supplemental conceptual notes for a compliant conditional DDPM; not an official assignment source and not code to copy.

Missing source documents: none of `doc/problem-brief.md`, `doc/repo-map.md`, or `doc/quality-gates.md` are missing. No package manifest, training code, generation code, checkpoint, or generated images exist yet.

## Current Project State

The repository is a sparse assignment workspace. It contains the assignment PDF, generated planning docs, dataset files, local scoring assets, and a minimal `README.md`, but no model implementation.

Observed gaps:

- No training script.
- No generation script.
- No model code.
- No dependency manifest.
- No validation script for generated images.
- No checkpoint.
- No final `generated_images/` output.
- No verified local scoring run.

The local scoring setup appears incomplete for full local evaluation because `scoring_program/input/ref/test.json` and reference test images were not observed. The scorer also uses 3,000 images in its config/manual, while the assignment requires 2,000 generated submission images.

## Problem Summary

The assignment requires generating Brainrot images conditioned on animal-object pairs. The dataset has 10 animal classes and 10 object classes, for 100 total combinations. Training uses 4,799 images and `train.csv`; generation uses 2,000 requested rows from `generate.csv`, with 20 requested images per pair.

Official grading uses:

- FID: lower is better, 50% of grade.
- CLIP-T: higher is better, 50% of grade, using prompts like `a shark and a sneaker`.

Final generated images must be stored as PNG files with filenames matching `generate.csv`.

## Constraints

- Train the main generator from scratch.
- Do not use pretrained generator weights.
- Do not use high-level pretrained diffusion or generation pipelines for the main model.
- Implement the main architecture, training loop, scheduler/loss, and sampling procedure directly.
- Pretrained CLIP or VAE may be used only as auxiliary modules, not as the final image generator.
- Preserve the required output: exactly 2,000 RGB PNG files at 64x64.
- Keep generated images, checkpoints, scorer outputs, and datasets out of source control unless explicitly packaging a submission.
- Document reproducible environment, training command, generation command, and checkpoint used.

## Proposed Approach

Implement a small conditional DDPM trained on `dataset/trainset/` and conditioned on `animal` and `object` labels from the CSV files.

Core approach:

- Use `dataset/train.csv` to map images to animal and object IDs.
- Train a from-scratch denoising model on 64x64 RGB images.
- Condition each denoising step on timestep, animal ID, and object ID.
- Sample images for every row in `dataset/generate.csv`.
- Save outputs to `generated_images/{id}`.
- Validate file count, filenames, format, color mode, and image size before scoring or submission.

The first implementation should prefer boring, reproducible Python/PyTorch code if approved, because the evaluator and coursework ecosystem already use Python and PyTorch. Avoid adding a larger framework unless training quality or reproducibility demands it.

## Algorithm Strategy

### Baseline Method

Start with a minimal class-conditional DDPM:

- Train objective: predict sampled Gaussian noise with MSE.
- Condition representation: separate embeddings for animal and object.
- Model shape: compact UNet-style denoiser with timestep and label conditioning.
- Sampler: standard reverse diffusion using the same schedule as training.
- Output: direct RGB image samples at 64x64.

The baseline succeeds when it can overfit a small subset and generate correctly named images for all 2,000 requested rows.

### Intended Optimized Method

After the baseline produces valid images, improve quality with low-risk additions:

- Exponential moving average weights for sampling.
- Balanced sampling by animal-object pair during training.
- Attention at low spatial resolutions if GPU memory allows.
- Tuned channel count, training length, learning rate, and diffusion steps.
- Optional classifier-free conditioning dropout only if it is implemented from scratch and improves CLIP-T/FID.

Avoid moving to latent diffusion, pretrained VAE generation, or copied research-framework code unless the baseline is exhausted and the assignment constraints are rechecked.

### Correctness Strategy

Correctness means the pipeline obeys the assignment interface and constraints:

- CSV parsing uses `animal` and `object` from `train.csv` and `generate.csv`.
- Generation writes exactly the filenames in `generate.csv`.
- Output validation checks exactly 2,000 PNG files.
- Every image is RGB and 64x64.
- A small overfit run confirms the training loop, conditioning, and sampler can learn.
- Fixed random seed and recorded config allow reproduction.

### Performance Strategy

Performance means improving both FID and CLIP-T without breaking reproducibility:

- Train on all 4,799 images after the small overfit check passes.
- Track per-pair sample grids during training to catch weak conditions.
- Use EMA weights for generation.
- Tune only a few high-impact settings first: base channels, training steps, batch size, learning rate, sampling steps, and balanced sampling.
- Run local scoring only after generated-image validation passes and the local scorer inputs are resolved.

## Alternatives Considered

- Conditional GAN: faster sampling, but likely less stable on a small 100-pair dataset and may require more tuning.
- Conditional VAE: simpler and reproducible, but direct samples may be blurrier and weaker for FID/CLIP-T.
- Autoregressive pixel model: compliant but likely too slow and complex for 64x64 RGB images.
- Transformer diffusion backbone: allowed if trained from scratch, but heavier than needed for the first working model.
- Latent diffusion with pretrained VAE: partially allowed as auxiliary use, but higher assignment-risk and unnecessary before a pixel-space DDPM baseline exists.
- Copying an external conditional diffusion repository: not acceptable as-is; external material may only inform ideas with license/provenance handled.

## Module Candidates

Keep the implementation surface small:

- `scripts/train.py`: train the conditional generator and save a checkpoint.
- `scripts/generate.py`: load a checkpoint and generate `generated_images/` from `dataset/generate.csv`.
- `scripts/validate_generated.py`: check file count, names, PNG/RGB mode, and 64x64 size.
- `scripts/model.py` or a small package module: model and diffusion utilities shared by train/generate.
- `requirements.txt`: minimal reproducible dependency list.
- `README.md`: environment, training, generation, validation, scoring notes, and checkpoint used.

This can be adjusted during high-level design, but the proposal should not expand into a large framework.

## Milestones

1. Establish environment and minimal scripts.
2. Implement dataset loading from `train.csv` and class mappings.
3. Build a small from-scratch conditional DDPM.
4. Overfit a small subset to verify training and sampling.
5. Train on the full dataset with recorded config and seed.
6. Generate all 2,000 requested images from `generate.csv`.
7. Validate generated files locally.
8. Run the local scorer if reference inputs and dependencies are available.
9. Update README and package reproducible submission assets.

## Validation Plan

Minimum validation gates:

- `train.csv` row count is 4,799 and image files exist for all rows.
- `generate.csv` row count is 2,000.
- Class mappings cover the 10 animals and 10 objects listed by the assignment.
- A small overfit run produces recognizable samples for a tiny subset.
- Generated output has exactly 2,000 files.
- Generated filenames exactly match `dataset/generate.csv`.
- Every generated image opens as PNG, RGB, 64x64.
- README records package versions, training command, generation command, seed, checkpoint path, and any extra data.

Optional scoring gate after setup is resolved:

```bash
cd scoring_program
python score.py --input_dir ./input --output_dir ./ --image_size 64 --num_images 3000 --test_json test.json --score fid clip_t clip_i --verbose
```

Treat local scores as diagnostic until the 2,000-vs-3,000 mismatch and missing local reference files are resolved.

## Risks and Tradeoffs

- A simple DDPM may need enough GPU time to beat the higher FID/CLIP-T bands.
- Pixel-space diffusion is straightforward but slower to sample than GANs or latent models.
- The small dataset may make overfitting easy; balanced sampling and visual per-pair checks reduce this risk.
- CLIP-T and FID may pull in different directions; tuning should inspect both metrics.
- Local scoring may not match Codabench until the reference setup is complete.
- Extra data is allowed but adds provenance and reproducibility burden.

## Assumptions

- The implementation will use Python and likely PyTorch unless the user chooses another stack.
- `dataset/train.csv`, `dataset/trainset/`, and `dataset/generate.csv` are the intended local inputs.
- `generated_images/` is the final Codabench upload source.
- No extra data will be used for the first version.
- The checkpoint can be named `model.pth` or another documented name accepted by the submission package.
- The first target is a valid, reproducible baseline before leaderboard tuning.

## Open Questions

- Which Python/CUDA/PyTorch environment should be used for training?
- What GPU and runtime budget are available?
- Should the first implementation use no pretrained auxiliary modules at all, or use CLIP only for local diagnostics?
- How should the local scoring setup be completed, given the missing `test.json`/reference images and the 3,000-image scorer setting?
- Will extra data be used? If yes, what source and license/provenance should be documented?
