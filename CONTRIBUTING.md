# Contributing

Each collaborator uses their own GPU, operating system, storage and cloud account.
GitHub is the shared code and review boundary. A shared Runpod Team or Network
Volume is not required. Earlier shared-Pod instructions describe historical setup.

## Branch ownership and review

- Keep `main` as the reviewed shared base. Start personal work on `feat/<owner>-<topic>`
  or a clearly owned existing feature branch.
- Never push, rebase, reset or force-push another collaborator's branch. Inspect it
  read-only when coordination is needed. Do not merge another person's work without
  that review being explicitly requested.
- Submit a focused PR with its goal, changed behavior, environment and validation.
  Environment-specific work remains on its branch until reviewed.
- Use a separate branch for changes to shared conventions or APIs. Do not bundle
  personal Runpod infrastructure into another collaborator's Windows setup.
- Prefer small additions over path renames while concurrent branches exist. Shared
  path moves require coordination with owners of affected branches.

## Locations and portability

See [repository layout](docs/repository_layout.md) and
[experiment record example](configs/experiment-record.example.json).
Use configurable dataset/output roots rather than requiring `/workspace`, a drive
letter or a developer's home directory in reusable code. Platform-specific launchers
can keep platform paths as explicit defaults inside their own environment directory.
A Linux CUDA package lock is not a universal Windows/Conda environment.

Never commit datasets, checkpoints, credentials, local virtual environments or large
prediction caches. Store local machine configuration under ignored `.local/` or an
explicit ignored environment file. Publish only small, non-secret reproducibility
records and summaries when they are needed by collaborators.

## Experiment comparability

Record dataset version and official split, selected sample tokens, subset scope,
checkpoint hash, code commit, resolved config, preprocessing, seed, GPU/driver/toolkit
and outputs. Distinguish mini, a trainval subset and full validation. Do not infer
full-dataset accuracy from a smoke test. Treat schema/protocol changes as shared work;
raw predictions and a particular cloud provider are local implementation choices.

Compute ownership stays with each member. A person's shutdown mechanism or budget
policy cannot be assumed to apply to another member's GPU. Shared scripts must not
create a paid instance as an import-time side effect.
