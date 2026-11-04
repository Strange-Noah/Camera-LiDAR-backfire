# Preservation and verification acceptance

- All 416 source working-tree files match their recovered copies by SHA256, including untracked reproducibility evidence.
- All 505 Git metadata files matched byte-for-byte immediately after copying, before the new recovery commit. All local refs and history were preserved.
- All 122 entries in the published rotation artifact hash manifest passed verification.
- All 240,351 pre-existing canonical workspace files passed comparison against the pre-copy SHA256 snapshot. No conflicting file was overwritten. Original repo refs and status were unchanged at that checkpoint.
- Documents sources were rehashed after copying and remained unchanged. They have not been deleted.
- AGENTS.md was added to the canonical workspace, both existing repository roots and the recovered repository. These new instruction files are intentional additions; prior experiment files remain untouched.
- No GPU experiment, Runpod API action, resource allocation, inference or paid compute was initiated. Migration used local file and Git operations.
- Git object integrity is checked before publication. This is storage/history acceptance, not a rerun of historical experiments or a claim that external runtime services have been retested.

[Machine-readable verification](verification.json). Full pre-copy canonical hash snapshot and original Git state are retained locally in the canonical `reports/workspace-recovery-2026-11-04` directory. Raw intermediate evidence remains local; the new commit contains rules and acceptance reports only.
