# Optional Local Scoring Adapter

## Goal

Keep the provided local scoring workflow documented and isolated so it can be used only after generated-output validation and scorer setup are complete.

## Inputs

- `doc/proposal.md`: Local scoring is optional and diagnostic until setup mismatch is resolved.
- `doc/high-level-design.md`: Scoring adapter is separate from core generation and validation.
- `doc/detailed-design.md`: Follow known scorer command, missing-input checks, and no-verified-claim policy.
- `doc/test-plan.md`: Label scorer commands as known/not run and keep local scorer setup issues explicit.

## Write Scope

README scoring section, optional scoring setup notes, optional lightweight setup-check script only if needed; no edits to official scorer unless explicitly requested.

## Read Scope

`scoring_manual.txt`, `scoring_program/score.py`, `scoring_program/input/ref/`, `doc/quality-gates.md`, validated generated output directory.

## Dependencies

Generated Output Validator; scorer reference files; scorer dependencies; user approval before running scorer.

## Tasks

- [x] Document the known local scorer command as known/not run until actually verified.
- [x] Add setup checks or README notes for required `test.json`, reference images, generated results directory, dependencies, and GPU expectations.
- [x] Keep the 2,000-vs-3,000 generated-image mismatch explicit.
- [x] Document that local scoring runs only after validator pass and explicit user approval.
- [x] Ensure scorer outputs such as `scores.json` remain out of source changes unless packaging/reporting is requested.

## Tests and Quality Gates

- [x] Scoring documentation does not claim evaluator verification before a real run.
- [x] Missing local scorer reference inputs are reported as setup-incomplete.
- [x] Known scorer command remains copied exactly from `doc/quality-gates.md`.

## Done When

- [x] Local scoring is documented as optional and isolated from core validation.
- [x] Setup limitations are explicit.
- [x] No scorer command has been run or marked verified by this task alone.

## Evidence

- Added `scripts/check_scoring_setup.py` to check scorer file layout without running the scorer.
- Added `tests/test_scoring_setup.py` for missing inputs, count-mismatch warning, and incomplete CLI status.
- README documents the known scorer command as not run and keeps local scoring diagnostic.
- Passed: `python -m unittest tests.test_scoring_setup` (3 tests).
- Passed: `python -m unittest discover -s tests -p 'test_*.py'` (30 tests).
- Real setup check reports incomplete as expected: missing `test.json`, missing reference image directory, missing `generated_images`, and `num_images=3000` vs expected 2,000 warning.
