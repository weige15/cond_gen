# Problem Brief

## Source Documents

- `HW6-Brainrot Image Generation.pdf` - read via `pdftotext`; primary assignment source for objective, dataset, constraints, evaluation, and submission format.
- `dataset/train.csv` - read; local training metadata with `id,animal,object` columns and 4,799 data rows.
- `dataset/generate.csv` - read; local generation metadata with `id,animal,object,prompt` columns and 2,000 data rows.
- `dataset/trainset/` - inspected; local training image directory contains 4,799 PNG files.
- `scoring_manual.txt` - read; local scoring setup and example command.
- `scoring_program/score.py` - read in relevant sections; local evaluator CLI and image validation behavior.
- `README.md` - read; currently only identifies the repository as `cond_gen`.
- `conditional_score_diffusion_heuristics.md` - read; supplemental implementation notes, not an official assignment source.
- `AGENTS.md` - read; repository guardrails and previously verified local context.

## Assignment Objective

Train a from-scratch conditional image generation model for Brainrot images conditioned on animal-object pairs. Use the trained model to generate 2,000 RGB PNG images at 64x64 resolution according to `dataset/generate.csv`.

## Required Inputs

- Training images from `dataset/trainset/`.
- Training metadata from `dataset/train.csv`, with columns `id`, `animal`, and `object`.
- Generation metadata from `dataset/generate.csv`, with columns `id`, `animal`, `object`, and `prompt`.
- `generate.csv` prompts use the form `a {animal} and a {object}`.
- The animal classes listed in the PDF are `shark`, `crocodile`, `frog`, `cat`, `dog`, `capybara`, `elephant`, `bird`, `fish`, and `monkey`.
- The object classes listed in the PDF are `sneaker`, `airplane`, `coffee cup`, `banana`, `cactus`, `toilet`, `pizza`, `drum`, `car`, and `chair`.
- Optional extra training data is allowed by the PDF, but the assignment does not guarantee it will improve final performance.
- Local scoring, if used, requires the scoring program and reference resources arranged under `scoring_program/input/ref` and generated images under `scoring_program/input/res`.

## Required Outputs

- Exactly 2,000 generated PNG images.
- Images must be 64x64 and RGB.
- Output filenames must match the `id` values in `dataset/generate.csv`.
- The assignment PDF says generated images should be stored under `generated_images/` for Codabench upload.
- The local scoring manual expects generated results under `scoring_program/input/res` when running the local evaluator.

## Constraints

- The main generator must be trained from scratch.
- The generator must not use pretrained generator weights, including pretrained UNet, Transformer, diffusion model, or other generative model weights.
- Students must implement the model architecture and training process themselves; the PDF explicitly disallows using high-level ready-made generation pipelines or training flows such as those from `diffusers` for the main model.
- Pretrained VAE or pretrained CLIP may be used only as auxiliary modules, such as for latent representation, feature extraction, or evaluation.
- Auxiliary pretrained modules must not replace the main generator or directly generate final images.
- The assignment does not restrict condition design; examples in the PDF include class embeddings, pretrained CLIP embeddings, or another custom condition mechanism.
- Training flow and trained model must be reproducible by the teaching assistants.
- The Brainrot Dataset is for this course assignment only and should not be reused for other purposes.
- Codabench submissions are limited to 3 uploads per day.
- Violating assignment restrictions, unreproducible results, not participating in the competition, or plagiarism can lead to a zero score according to the PDF.

## Evaluation or Grading Criteria

- Official evaluation uses FID and CLIP-T; each contributes 50% of the assignment score.
- FID is better when lower and measures distribution distance from the hidden test set.
- CLIP-T is better when higher and measures semantic agreement between generated images and text prompts.
- The hidden test set has 3,000 images, with 30 images per animal-object pair.
- FID score bands from the PDF:
  - `<= 49.2545`: 100% of the FID portion.
  - `<= 58.0755`: 90% of the FID portion.
  - `<= 75.0642`: 80% of the FID portion.
  - `<= 90.0142`: 70% of the FID portion.
  - Otherwise: 0% of the FID portion.
- CLIP-T score bands from the PDF:
  - `>= 0.2703`: 100% of the CLIP-T portion.
  - `>= 0.2618`: 90% of the CLIP-T portion.
  - `>= 0.2536`: 80% of the CLIP-T portion.
  - `>= 0.2170`: 70% of the CLIP-T portion.
  - Otherwise: 0% of the CLIP-T portion.
- The PDF states CLIP-T is computed with OpenAI CLIP `ViT-B-32-quickgelu`.
- The local scoring script defaults to `ViT-B-32-quickgelu` with pretrained weights `openai`.
- The local scoring manual gives this example command:

```bash
cd scoring_program
python score.py --input_dir ./input --output_dir ./ --image_size 64 --num_images 3000 --test_json test.json --score fid clip_t clip_i --verbose
```

- The local scorer writes `scores.json` to the selected output directory.
- The PDF lists deductions for late submission, incorrect file format, plagiarism, unreproducible results, assignment-rule violations, and non-participation.

## Required Deliverables

- Upload generated images to Codabench by zipping the contents of `generated_images/`.
- Submit an E3 package named `HW6_{student_id}.zip`.
- The PDF example package structure is:

```text
HW6_{student_id}/
├── generated_images/
├── scripts/
├── model.pth
├── README.md
└── requirements.txt
```

- `README.md` should clearly describe environment setup, training, and generation commands so the teaching assistants can reproduce the result.
- If extra data is used or model weights exceed the E3 upload limit, the PDF allows providing a cloud link inside the submission materials.

## Relevant Methods From Papers

None found in the provided sources.

## Data, Benchmarks, or Test Cases

- The Brainrot Dataset contains animal-object pair images gathered from public web data and filtered for the assignment.
- The dataset covers 10 animal classes and 10 object classes, for 100 total animal-object combinations.
- The PDF says the training set contains 4,799 images.
- Local `dataset/train.csv` has 4,799 data rows across 10 animals, 10 objects, and 100 animal-object pairs.
- Local `dataset/trainset/` contains 4,799 PNG files.
- Local `dataset/generate.csv` has 2,000 data rows across 100 animal-object pairs.
- Local `dataset/generate.csv` has exactly 20 rows per animal-object pair.
- Local `dataset/train.csv` pair counts range from 22 to 60 images per animal-object pair.
- The hidden test set described by the PDF has 3,000 images, with 30 images per animal-object pair.
- `hw6_reference.zip` and `scoring_program.zip` are present locally, but the reference zip was not unpacked during this brief.

## Implementation Environment

- Python is present and used by `scoring_program/score.py`.
- The local scoring program imports `numpy`, `open_clip`, `torch`, `PIL`, `torchvision`, `tqdm`, and `scipy`.
- The main project toolchain and package manager are not established in the provided sources.
- The PDF baseline table mentions an RTX 4070 with 12GB VRAM for baseline training environments, but does not require that exact hardware.
- The repository `README.md` currently does not document environment setup, training, or generation commands.

## Confirmed Facts

- The named assignment PDF exists locally and was read.
- The assignment requires a conditional image generation model trained from scratch.
- Final generated images must be based on `dataset/generate.csv`.
- Final generated images must use the requested filenames from `generate.csv`.
- `dataset/generate.csv` contains 2,000 generation requests.
- `dataset/train.csv` contains 4,799 training metadata rows.
- `dataset/trainset/` contains 4,799 PNG files.
- Official grading uses FID and CLIP-T, weighted 50% each.
- The assignment disallows pretrained generator weights and high-level pretrained generation pipelines for the main model.
- The assignment allows pretrained VAE or CLIP only as auxiliary modules.
- The training process and final model must be reproducible.

## Assumptions

- `dataset/train.csv`, `dataset/generate.csv`, and `dataset/trainset/` are the intended local training and generation inputs for this workspace.
- `generated_images/` should contain only the 2,000 final generated PNG files for Codabench packaging.
- The student ID placeholder in `HW6_{student_id}.zip` will be replaced before final submission.
- Since no project training code exists yet, future implementation can choose the minimal compliant architecture and training setup as long as it preserves assignment constraints.

## Open Questions

- The exact E3 assignment deadline is not visible in the extracted PDF text.
- The local scoring manual example uses `--num_images 3000`, while the assignment PDF and local `generate.csv` require 2,000 final generated images. Confirm the intended local scoring setup before treating local scores as final-submission-equivalent.
- The package manager, Python version, CUDA version, and final training hardware are not specified.
- The expected checkpoint filename is shown as `model.pth` in the PDF example, but the assignment may accept another clearly documented filename if README instructions are complete.
- It is unknown whether extra data will be used; if it is, provenance and reproducibility details must be documented.

## Notes for Proposal Generation

- Preserve the assignment constraints on pretrained weights and high-level generation libraries.
- Start any implementation proposal from the confirmed CSV schemas, file counts, output filenames, and 64x64 RGB PNG requirement.
- Keep implementation design separate from this brief; the PDF permits multiple condition designs and does not mandate a specific architecture.
- Include an explicit validation step for generated file count, filenames, image mode, and image size before scoring or packaging.
- Treat the 2,000-vs-3,000 local scoring mismatch as proposal-impacting until the local evaluation workflow is confirmed.
