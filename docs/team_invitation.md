# Message for teammates

We are organizing Camera–LiDAR-backfire around a shared GitHub repository and a
Runpod Team. Please send me the email address associated with your own Runpod
account so I can invite you with the **Dev** role. Use your own login, credentials
and SSH key; never share passwords, API keys or private SSH keys.

Each person will create and manage a separate Pod **inside the Team account**, in
EU-RO-1, mounting our existing Network Volume at `/workspace`. Team usage is charged
to the shared Team balance, not your personal Runpod balance. We will attribute
costs by member using Cost Centers and settle them internally.

Clone https://github.com/Strange-Noah/Camera-LiDAR-backfire into your own Pod-local
home directory, create a branch named `<github-login>/<task>` from current `main`,
and commit/push to that branch. On a replacement Pod, clone again and check out the
same remote branch. Keep your working tree and Python environment local to your
Pod. Open a pull request when your work is ready; merge into `main` only after team
review and agreement. Do not directly push to main, casually merge other people's
branches, or force-push shared branches.

The shared dataset is `/workspace/datasets/nuscenes`. Treat it strictly as input:
never overwrite, move or delete its files, and never save derived data beside the
originals. Use the agreed non-root experiment login and wait for the administrator
to confirm read-only permissions are in place; **Dev membership alone does not
protect the dataset**. Write results to
`/workspace/results/<github-login>/<experiment>/<unique-run-id>/` and use personal
cache/staging directories. Preserve the published `/workspace/results/metadata_audit`.
Do not run experiments as root or use broad recursive permission changes.

Dev access also permits deleting shared storage. Never delete or reconfigure the
Network Volume, or manage another member's Pod, without agreement. Stop your own
idle Pod, push code before shutdown, and persist needed outputs in your own run
directory. See `docs/team_collaboration.md` for the setup commands and safety rules.
The Team/access and filesystem-protection rollout is pending confirmation; do not
assume either is already active merely because these instructions are in GitHub.
