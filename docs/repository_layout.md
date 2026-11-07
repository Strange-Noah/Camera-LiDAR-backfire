# Repository layout for independent compute

The existing top-level layout is sufficient; no source relocation is needed.
This document proposes conventions for review without moving existing files.

| Path | Shared purpose | Placement rule |
|---|---|---|
| `src/` | Reusable model/data/evaluation logic | Keep research APIs independent of cloud and machine paths |
| `scripts/` | Explicit command-line entry points | Label platform requirements; no implicit paid provisioning |
| `configs/` | Small parameter and experiment examples | Roots are supplied per machine; no secrets |
| `environment/<workflow>/` | Isolated reproducible environments | Keep metadata, BEVFusion/Linux and local Windows dependencies separate |
| `docs/` | Methods and shared workflow documentation | Separate historical deployment assumptions from current collaboration |
| `tests/` | Meaningful correctness and contract checks | CPU checks run before optional GPU acceptance |
| `results/<experiment>/` | Small reviewed summaries and manifests | Large artifacts remain outside Git; record persistent location |
| `.local/` | Ignored machine-specific state | Credentials, private reports, SSH metadata and worktrees stay local |

New independent contributions can add a focused `src/<topic>/` module, a namespaced
configuration, a documented launcher and small validation evidence. Avoid creating a
complete copy of the repository per experiment. Reuse common dataset/record formats
when validated; share an interface through a separate reviewed PR when it stabilizes.

## Existing concurrent branches (read-only inventory, 2026-10-20)

- `main`: public base, `bd6625e292995418e96490dd001dd7f3e01af4a9`.
- `refactor/windows-local-setup`: teammate yilong's Windows/Conda work,
  `a550abcbf0e8ee345bdd57ba2b78dd88441f44d2`; one commit ahead of the base.
- `feat/bevfusion-mini-deployment`: owner's BEVFusion/Runpod work, reviewed separately.

This proposal adds documentation and a record example only. It does not modify,
merge or relocate either feature branch's files. Historical Runpod Team setup in
README/docs is not a prerequisite for current independent-compute collaboration.

## Evolving main

Keep platform-specific work and new experiment implementations in their feature
branches until reviewed. Land stable shared contracts and portable helpers through
small separate PRs. When a shared path must change, coordinate with affected branch
owners and preserve compatibility entry points before removing old ones.
