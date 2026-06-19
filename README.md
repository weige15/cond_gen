# HW6 Brainrot Conditional Image Generation

This repository implements a from-scratch label-conditioned DDPM for HW6.
It trains on `dataset/train.csv` and `dataset/trainset/`, then generates RGB 64x64
PNG files for every row in `dataset/generate.csv`.

## Environment

Observed local environment:

- Python 3.12
- PyTorch 2.9.0+cu128
- NumPy 2.2.6
- Pillow 12.2.0

Install the minimal Python dependencies:

```bash
python -m pip install -r requirements.txt
```

## Data Check

Validate assignment CSVs and training image files before training:

```bash
python scripts/train.py --train_csv dataset/train.csv --image_dir dataset/trainset --data_check_only --full_data_check
```

For a stricter image decode check:

```bash
python scripts/data.py --train_csv dataset/train.csv --generate_csv dataset/generate.csv --image_dir dataset/trainset --full --decode_images
```

## Train

Train the from-scratch conditional DDPM and write `model.pth`.
This A100-oriented recipe uses a wider UNet, residual depth, low-resolution
self-attention, cosine noise schedule, classifier-free condition dropout, AMP,
gradient clipping, and EMA weights:

```bash
python scripts/train.py --train_csv dataset/train.csv --image_dir dataset/trainset --checkpoint model.pth --seed 0 --full_data_check --steps 150000 --batch_size 128 --lr 2e-4 --device auto --base_channels 128 --channel_mults 1,2,3,4 --num_res_blocks 2 --attention_resolutions 16,8 --attention_heads 4 --cond_dim 512 --dropout 0.1 --condition_dropout 0.1 --timesteps 1000 --schedule cosine --ema_decay 0.9999 --grad_clip 1.0 --amp
```

The implementation pass did not run full training or create a final checkpoint.
Use the exact checkpoint path passed to `--checkpoint` for generation.

## Generate

Generate the 2,000 requested images:

```bash
python scripts/generate.py --checkpoint model.pth --generate_csv dataset/generate.csv --output_dir generated_images --seed 0 --batch_size 128 --sampling_steps 250 --guidance_scale 1.5 --ddim_eta 0.0 --use_ema
```

`generated_images/` must be empty or nonexistent. Use `--overwrite` only to
replace files whose names are requested by `dataset/generate.csv`; unrelated
extra files still fail.

## Validate Output

Run this before scoring or packaging:

```bash
python scripts/validate_generated.py --generate_csv dataset/generate.csv --image_dir generated_images --image_size 64 --expected_count 2000
```

The validator checks exact filenames, exact count, PNG format, RGB mode, and
64x64 size.

## Tests

Run the project-owned checks:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## Assignment Compliance

- Main generator is trained from scratch.
- No pretrained generator weights are used.
- No `diffusers` pipeline or high-level generation framework is used.
- Animal and object labels come from the assignment CSV files.
- No extra data is used by default.
- Checkpoints, generated images, scorer outputs, datasets, and archives should
  stay out of git unless explicitly packaging a submission.

## Local Scoring Caveat

Known local scorer command from the provided materials, not run by this
implementation pass:

```bash
cd scoring_program
python score.py --input_dir ./input --output_dir ./ --image_size 64 --num_images 3000 --test_json test.json --score fid clip_t clip_i --verbose
```

Treat local scoring as diagnostic until generated images exist, required scorer
reference files are present, dependencies/GPU are available, and the 2,000 vs
3,000 image-count mismatch is resolved.

Check local scorer file layout without running the scorer:

```bash
python scripts/check_scoring_setup.py --scoring_dir scoring_program --generated_dir generated_images --expected_count 2000
```
