# AGENTS.md

## Project Context

- Project: HW6 Brainrot Image Generation.
- Assignment goal: train a from-scratch conditional image generation model and
  generate 2,000 RGB PNG images at 64x64 using `dataset/generate.csv`.
- Local assignment files verified: `HW6-Brainrot Image Generation.pdf`,
  `dataset/train.csv`, `dataset/generate.csv`, `scoring_manual.txt`,
  `scoring_program/score.py`.
- Repository remote provided by user: `git@github.com:weige15/cond_gen.git`.
- Current checkout status during setup: this directory was not a valid git
  repository; `.git/` existed but was empty.
- Language/toolchain: Python is present in the scoring program; project model
  implementation toolchain is otherwise unknown.
- Package manager: Unknown.

## Repository Rules

- Preserve assignment constraints: no pretrained generator weights, no
  high-level pretrained diffusion/generation pipelines for the main model, and
  no copied repository code submitted without proper license handling.
- Keep generated training artifacts, checkpoints, and generated image folders
  out of source changes unless the user explicitly asks to package submission
  files.
- Do not overwrite dataset, PDF, zip, scoring reference, or generated results
  without explicit permission.
- Prefer small, reviewable changes. Do not refactor unrelated files.

## Read-Only Discovery Commands

These commands are safe for future agents to run without permission:

```bash
pwd
ls -la
find . -maxdepth 2 -type f
rg --files
git status --short --branch
git branch --show-current
git worktree list
sed -n '1,220p' AGENTS.md
sed -n '1,220p' scoring_manual.txt
```

If git commands report that this is not a repository, do not repair git state
without permission.

## Commands Requiring Permission

Ask before running commands that create, edit, move, delete, or upload anything,
including:

- editing or creating source/docs/config files
- installing dependencies
- running training, generation, scoring, build, lint, format, or test commands
- creating large outputs such as checkpoints or `generated_images/`
- initializing git, adding remotes, changing branches, committing, pushing,
  rebasing, merging, stashing, or applying patches

## Forbidden Commands

Do not run these unless the user explicitly requests the exact operation:

```bash
rm -rf
git reset --hard
git clean -fd
git checkout -- .
git restore .
git push --force
git push --force-with-lease
chmod -R
chown -R
sudo
```

## Build, Test, and Quality Gates

- Verified local scoring command from `scoring_manual.txt`:

```bash
cd scoring_program
python score.py --input_dir ./input --output_dir ./ --image_size 64 --num_images 3000 --test_json test.json --score fid clip_t clip_i --verbose
```

- Running the scoring program requires permission because it writes outputs such
  as `scores.json` and may load/download model weights depending on environment.
- Training command: Unknown.
- Generation command: Unknown.
- Lint command: Unknown.
- Format command: Unknown.
- Type-check command: Unknown.
- Test command besides scoring program: Unknown.

## Documentation Rules

- Keep `README.md` focused on reproducibility: environment, training command,
  generation command, checkpoint used, and package versions.
- Record any use of external references or implementation ideas. Do not claim
  copied code as original work.
- Do not store secrets, tokens, private paths, or credential values in docs.
- If documenting generated outputs, include file counts, image size, and command
  used to produce them.

## Coding Rules

- Prefer a minimal HW6-specific implementation before adapting large research
  frameworks.
- Keep condition handling explicit: animal id and object id should come from the
  CSV files.
- Ensure generation writes exactly the filenames requested by
  `dataset/generate.csv`.
- Validate image outputs as RGB PNG, 64x64.
- Keep model code, scripts, and configs separate from downloaded datasets,
  archives, scoring references, checkpoints, and generated images.

## Git and Commit Rules

- Current directory was not a valid git repository during guardrail setup. Ask
  before initializing git, cloning, or connecting to the provided remote.
- Before committing, inspect `git status --short` and avoid staging unrelated
  user changes.
- Do not push to `git@github.com:weige15/cond_gen.git` without explicit user
  permission.
- Commit messages should be concise and task-specific, for example:
  `docs: add project guardrails`.

## Uncertainty Protocol

- Treat unverified facts as unknown and say so.
- If assignment rules conflict with a library, repo, or shortcut, follow the
  assignment rules.
- If a command could overwrite data, create large files, consume significant GPU
  time, or alter git state, ask first.
- When unsure whether generated results are valid, check filename count, image
  size, color mode, and local scoring setup before submission packaging.

