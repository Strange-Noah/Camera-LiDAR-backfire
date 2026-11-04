# Camera–LiDAR Backfire agent operating rules

## Startup verification — before filesystem modifications

1. Read this AGENTS.md and all applicable ancestor instructions.
2. Confirm `/Users/strangenoah/Desktop/AI+X/Visual_Computing/Camera_LiDAR_backfire` exists; resolve its actual canonical path and any symlinks. Never create a substitute because a name differs.
3. Resolve the working directory and every intended write destination; verify they remain under that canonical root.
4. Verify the Git repository, remotes, active branch and worktree status. Use the user's personal branch and preserve unrelated work.
5. Confirm the requested action does not initiate unauthorized GPU spending or production changes.
6. Stop if workspace identity or permission boundaries cannot be verified.

## Authorized workspace

The sole authorized local working root is:
`/Users/strangenoah/Desktop/AI+X/Visual_Computing/Camera_LiDAR_backfire`

Keep routine project work, isolated Git worktrees, test outputs, reports and temporary files under this root wherever technically feasible. Do not establish working directories under `~/Documents/ChatGPT` or Google Drive-backed locations. Resolve actual paths and symlinks before writing. Any required access outside the root must be justified and explicitly approved; existing task-specific approval applies only to its stated scope. Recovery copies remaining under Documents are retained evidence, not active workspaces. Do not delete them or change synchronization settings without authorization.

These are instruction-level operating rules. They do not change or override actual filesystem permissions. A session may enforce narrower writable boundaries; obtain the required platform permission before writing outside those boundaries. AGENTS.md cannot grant access, reconfigure the app's workspace or enforce a filesystem restriction by itself.

## Experiment integrity

- Preserve accepted baselines, experiment outputs, non-reproducible predictions, metadata, manifests and acceptance evidence. Do not assume paid experiments can be rerun.
- Never silently replace outputs produced in different environments or overwrite conflicting contents.
- Record dataset versions, model and evaluation configurations, seeds, error distributions, run conditions, source/runtime identities and metrics.
- Keep clean, spatial/rotation misalignment and temporal misalignment experiments distinct.
- Define fusion backfire relative to the accepted LiDAR-only baseline. Distinguish observed grid crossings, interpolation and statistically supported claims; keep mAP and NDS conclusions separate.
- Report all validation failures and important numerical discrepancies. Present only verified experiments as accepted results.

## GPU and cost authorization

- No additional GPU expenditure without explicit approval. Do not restart Pods or repeat paid inference merely to regenerate existing outputs.
- Reuse validated results where appropriate; preserve cost and execution records.
- Do not expose API keys or credentials.
- Maintain bounded execution and safe resource cleanup for authorized runs.

## GitHub collaboration

- Work on the user's personal branch unless explicitly authorized otherwise. Do not merge into main or teammates' branches without approval.
- Preserve branches, remotes and history. Do not overwrite or rewrite teammates' commits; do not force-push or rewrite history.
- Upload appropriate final artifacts and reproducible implementations. Keep raw intermediate work and local recovery evidence separate from deliverable code.
- Describe deliverables by completed outcomes. Preserve validation issues in the appropriate acceptance records.
- Maintain GitHub reports and acceptance records; provide direct GitHub links to summary, verification and handoff documents.
