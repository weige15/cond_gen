# Conditional Score Diffusion Heuristics for HW6

Source repository: https://github.com/GBATZOLIS/conditional_score_diffusion

This note extracts design heuristics from the repository without treating it as a
drop-in solution. For HW6, the generator must be trained from scratch, must be
reproducible, and must not use pretrained generator weights or high-level
diffusion pipelines.

## Compliance Boundaries

- Do not use the repository's checkpoints, hardcoded checkpoint paths, or any
  pretrained generative weights.
- Do not submit copied code as-is. If you reuse implementation details, cite the
  source and keep license headers where required.
- Do not use `diffusers` pipelines or other high-level pretrained generation
  workflows for the main model.
- It is acceptable to use the repository as a conceptual reference for UNet
  layout, noise schedules, score matching losses, and sampler structure.
- For HW6, replace the repo's paired image condition `y` with animal/object
  conditioning derived from `train.csv` and `generate.csv`.

## Useful Heuristics

### 1. Keep the First Working Baseline Simple

The repository is large because it supports several research settings. HW6 does
not need that full framework. A small conditional DDPM is the practical baseline:

- input: noisy 64x64 RGB image
- conditions: animal id and object id
- output: predicted noise or score
- training loss: MSE between predicted and sampled noise
- sampling: reverse diffusion from Gaussian noise

This covers the assignment target without inheriting inverse-problem machinery.

### 2. Use a UNet with Time and Class Conditioning

The repo's DDPM/NCSN++ models use residual UNet blocks, timestep embeddings,
GroupNorm, Swish/SiLU activations, and attention at lower resolutions. For HW6:

- embed timestep with sinusoidal or learned embeddings
- embed animal and object separately
- combine condition embedding with timestep embedding
- inject the combined embedding into residual blocks
- use attention at 16x16 or 8x8 if GPU memory allows

Small starting point: base channels 64 or 96, channel multipliers
`(1, 2, 2, 4)`, two residual blocks per level.

### 3. Treat Animal and Object as Separate Conditions

The assignment condition is compositional: 10 animals x 10 objects. Do not reduce
it to one opaque class id unless the simple baseline already works.

Recommended conditioning:

- `animal_embedding[10]`
- `object_embedding[10]`
- optional pair embedding `[100]` as an extra signal
- combined condition = timestep embedding + animal embedding + object embedding

This keeps the model aware that "dog + banana" shares information with other
dog images and other banana images.

### 4. Adapt the Repo's Conditional Idea, Not Its Interface

The repo conditions on image-like tensors for tasks such as super-resolution,
inpainting, and image-to-image translation. HW6 conditions on labels/prompts.

Use the idea "condition the score model at every denoising step," but implement
it as label conditioning. Do not force labels into fake image tensors just to
match the repo's paired-dataset API.

### 5. Start With DDPM Before Score-SDE Complexity

The repo includes VE/VPSDE machinery and predictor-corrector samplers. For HW6,
a standard DDPM or DDIM-style implementation is easier to reproduce:

- fixed beta schedule, e.g. linear or cosine
- 1000 training timesteps
- train by randomly sampling one timestep per image
- sample with 1000 steps first, then try fewer steps only after quality is sane

Only add predictor-corrector or VE-SDE sampling if the basic DDPM is already
stable and time remains.

### 6. Use EMA for Sampling Weights

The repository includes EMA support. This is a low-risk quality improvement:

- keep an exponential moving average of model weights
- sample with EMA weights, not raw training weights
- try EMA decay around `0.999` or `0.9999`

This usually improves FID without changing the training objective.

### 7. Match the Scoring Interface Early

HW6 grading requires exactly 2,000 PNG files matching `generate.csv`.

Build generation around:

- read `dataset/generate.csv`
- for each row, use animal/object ids as condition
- save `generated_images/{id}`
- ensure output is RGB PNG, 64x64

Do this before chasing model improvements, because wrong filenames or counts
can waste a leaderboard submission.

### 8. Optimize Against Both Metrics

FID and CLIP-T can pull in different directions:

- FID prefers realistic, diverse samples close to the hidden test distribution.
- CLIP-T prefers semantic match to `a {animal} and a {object}`.

Practical heuristic:

- train on all labels with balanced sampling by animal/object pair
- inspect per-pair samples, not only aggregate FID
- avoid overfitting to a tiny subset of visually easy pairs
- use the local scoring program after every meaningful generation run

### 9. Borrow Config Values, Not Config Structure

The repo's configs are useful for scale hints but are too heavy for HW6:

- batch size depends on GPU memory; use the largest stable batch
- Adam with learning rate around `2e-4` is a reasonable starting point
- gradient clipping around `1.0` can prevent unstable updates
- warmup can help early training stability
- 500k steps is a research setting, not a requirement; start smaller and measure

### 10. Keep Reproducibility Boring

The assignment says the training process and trained model must be reproducible.
Record:

- exact command for training
- exact command for generation
- model architecture parameters
- random seed
- number of steps/epochs
- optimizer and learning rate schedule
- checkpoint used to generate final images
- Python package versions

Put this in your final `README.md`.

## Minimal Experiment Plan

1. Build a tiny conditional DDPM and verify it overfits a 100-image subset.
2. Train on all 4,799 images until samples become recognizable.
3. Generate all 2,000 images using `generate.csv`.
4. Run the local scoring program.
5. Add EMA if not already present.
6. Tune condition injection, channel count, training steps, and sampling steps.

## What to Avoid

- Adapting the full PyTorch Lightning research framework before having a working
  HW6 generator.
- Using paired image-to-image datasets or fake `A/B` folders for label
  conditioning.
- Depending on the repo's pretrained checkpoints.
- Spending effort on multi-scale Haar or inverse-problem estimators before the
  basic class-conditional generator works.
- Reporting only training loss; always inspect generated images and local FID /
  CLIP-T.

