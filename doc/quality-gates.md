# Quality Gates

## Environment Summary

- Repository root: `/home/kuotzuwei15/GenAI/hw6`.
- Current project implementation is not present yet: no model code, training script, generation script, package manifest, or dependency lockfile was discovered.
- Python is used by the local scoring program in `scoring_program/score.py`.
- `dataset/train.csv` has 4,799 data rows plus header; `dataset/generate.csv` has 2,000 data rows plus header.
- `dataset/trainset/` contains 4,799 PNG images.
- `scoring_program/input/ref/` contains `test_mu.npy`, `test_sigma.npy`, and `config.json`.
- `scoring_program/input/ref/test.json` and `scoring_program/input/ref/test/` were not found.
- `scoring_program/input/res/` exists but contains 0 files.
- No quality command was run in this session; evaluator execution requires explicit approval because it writes `scores.json`, may use GPU, may load or download model weights, and currently lacks required local scorer inputs.

## Build Commands

- **Missing**: No build command was found in project metadata, docs, CI, task runners, or scripts.

## Unit Test Commands

- **Missing**: No unit test command or test framework configuration was found.

## Integration Test Commands

- **Missing**: No integration test command was found.

## Lint Commands

- **Missing**: No lint command or lint configuration was found.

## Format Commands

- **Missing**: No format command or formatter configuration was found.

## Type-Check Commands

- **Missing**: No type-check command or type-check configuration was found.

## Static Analysis Commands

- **Missing**: No static-analysis command was found.

## Benchmark or Evaluator Commands

- **Discovered** from `scoring_manual.txt` and `AGENTS.md`:

```bash
cd scoring_program
python score.py --input_dir ./input --output_dir ./ --image_size 64 --num_images 3000 --test_json test.json --score fid clip_t clip_i --verbose
```

  - Working directory: `/home/kuotzuwei15/GenAI/hw6/scoring_program`.
  - Writes: `scoring_program/scores.json`.
  - Likely requirements: populated `input/ref/test.json`, `input/ref/test/`, generated images in `input/res/`, Python dependencies, CUDA-capable PyTorch because `score.py` selects `cuda:0`.
  - Risk notes: may load or download OpenAI CLIP and Inception weights depending on environment; uses large image sets; may require GPU and nontrivial runtime.
  - Status: **Discovered**, not verified.

- **Discovered** from `scoring_program/metadata`:

```bash
python3 score.py --input_dir $input --output_dir $output --config config.json
```

  - Likely working directory: `/home/kuotzuwei15/GenAI/hw6/scoring_program` or an external scoring harness directory.
  - `$input` and `$output` are scoring-harness placeholders, not local shell values discovered in this repo.
  - `config.json` is resolved by `score.py` relative to `$input/ref/`.
  - Local `scoring_program/input/ref/config.json` sets `image_size: 64`, `num_images: 3000`, and scores `fid`, `clip_t`, `clip_i`.
  - Status: **Discovered**, not verified.

## Smoke Test Commands

- **Missing**: No smoke-test command was found for validating generated image count, filenames, RGB mode, or 64x64 size.

## Verified Commands

- None. No build, test, lint, format, type-check, static-analysis, benchmark, smoke-test, evaluator, package-manager, or project script command was run in this session.

## Commands Not Run

- `cd scoring_program && python score.py --input_dir ./input --output_dir ./ --image_size 64 --num_images 3000 --test_json test.json --score fid clip_t clip_i --verbose`
  - Reason: requires explicit approval; writes `scores.json`; likely needs GPU and model weights; local `input/ref/test.json`, `input/ref/test/`, and generated result images are absent.
- `python3 score.py --input_dir $input --output_dir $output --config config.json`
  - Reason: discovered as scoring-harness metadata with unresolved `$input` and `$output`; not a directly runnable local command without harness-provided paths.
- `python score.py --help`
  - Reason: documented helper command in `scoring_manual.txt`, but not a quality gate and not run.

## Missing Quality Gates

- Build gate: missing.
- Unit test gate: missing.
- Integration test gate: missing.
- Lint gate: missing.
- Format gate: missing.
- Type-check gate: missing.
- Static-analysis gate: missing.
- Smoke-test gate: missing.
- Project-owned generated-image validation gate: missing.
- Training and generation reproducibility commands: missing.

## Recommended Minimum Done Criteria

- Before implementation work is considered done, add documented training and generation commands to `README.md`.
- Add a minimal generated-output validation gate that checks exactly 2,000 files, filenames matching `dataset/generate.csv`, PNG format, RGB mode, and 64x64 size.
- Once local scoring inputs and generated images exist, run the discovered local evaluator command with approval and record the result.
- Resolve the mismatch between the assignment output requirement of 2,000 images and the local scorer config/manual value of `num_images: 3000` before treating local scores as final-submission-equivalent.
- Keep generated images, checkpoints, scoring outputs, and datasets out of source control unless explicitly packaging a submission.
