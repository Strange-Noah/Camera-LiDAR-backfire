# Team collaboration

## Agreed architecture and current rollout status

GitHub main is the reviewed project definition. Personal branches and Pod-local
checkouts are independent; the Network Volume is the shared large-data layer.
Use the existing owner's Runpod Team with Dev members, not Organization Beta.
This document does not establish that the account has already been converted.

Verified existing volume: `camera-lidar-backfire-data`, `p6vb29fmlu`, 50 GB STANDARD,
EU-RO-1. Existing dataset path: `/workspace/datasets/nuscenes`.
Team conversion, member access, non-root logins and dataset permission enforcement
remain pending live verification. Do not claim the dataset is already protected.
No Pod needs to run for Team conversion. Do not recreate or copy the dataset.

## Owner: enable Team and invite members

1. Sign in to the account that owns volume `p6vb29fmlu`.
2. Open Team; if still personal, select **Convert to a Team Account** and name it
   `Camera-LiDAR-backfire`. If already a Team, retain the existing Team.
3. Confirm the same volume ID remains visible in Team Storage with EU-RO-1 and 50 GB.
   Stop if it is absent; do not create a replacement or attempt a data migration.
4. Ask each teammate for the email associated with their own Runpod account.
5. Team -> Members -> Invite New Member -> Dev -> email -> Create Invite.
   Send each invitation link privately; never commit links or member email lists.
6. Each member signs in as themselves, accepts, and selects the Team context.
   Confirm they can see the existing volume in the deployment selector before
   launching paid compute. GitHub invitations are separate from Runpod invitations.

Traditional Team resources remain owned by the Team owner's account. Converting
that existing account is the intended in-place sharing route, not a storage move.
Do not join Organization Beta as a substitute for this workflow.

## Billing and Pod lifecycle

Dev members may create/start/stop/delete Pods and manage Network Volumes.
All Team-context resources charge the Team account, regardless of who connects.
Use one Cost Center per member and `shared-storage` for the common volume; settle
costs internally. Cost Centers are labels, not separate wallets or hard budgets.
Assign each new Pod promptly. Track Pod IDs as well as names for reconciliation.

Create compatible Secure Cloud Pods in **EU-RO-1**, attach `p6vb29fmlu` at `/workspace`
at creation, and size CPU/GPU for the actual task. A Git branch does not choose the
billing account: check the Team selector before deploying. Use personal credentials
and SSH keys, not the owner's password, API key or private key. Do not put secrets
on the shared volume. Push work before stopping a Pod; local checkouts and environments
are disposable. Checkpoint long-running work to your own persistent run directory.
Only manage your assigned Pods; do not stop or delete another member's resources.

## Personal branch and working tree

After the administrator prepares a non-root experiment login, connect as that user.
Clone into that user's local home, outside `/workspace`. Each Pod has its own clone:

```bash
git clone https://github.com/Strange-Noah/Camera-LiDAR-backfire.git ~/camera-lidar-backfire
cd ~/camera-lidar-backfire
git switch main
git pull --ff-only origin main
git switch -c <github-login>/<task>
git push -u origin HEAD
```

Replace angle-bracket placeholders before running commands. For an existing branch
on a replacement Pod, fetch then `git switch --track origin/<github-login>/<task>`.
Commit and push frequently. Open a PR into main after validation and team review;
do not directly push to main, merge someone else's branch without agreement, or
force-push shared branches. These are team conventions; branch protection has not
been configured by this rollout. Local branches are not backups until pushed.

Each experiment should record the commit SHA, configuration, dataset version and
run ID. Never commit datasets, weights, secrets, private download URLs or large
outputs. Small selected reports can be added to Git in a reviewed PR.

## Separate environments and outputs

```bash
mkdir -p .local
cp configs/member.env.example .local/member.env
# Edit MEMBER_SLUG to your GitHub login, then:
source .local/member.env
./scripts/setup_metadata_env.sh
./scripts/verify_nuscenes.sh
# Example only: run deliberately, with a fresh output directory each time.
./scripts/run_metadata_audit.sh --version v1.0-mini \
  --output "/workspace/results/$MEMBER_SLUG/mini-check/<unique-run-id>"
```

The administrator must provision writable per-member directories before these
commands. Re-source the member file in every new shell. Use UTC time plus a UUID
for run IDs, and refuse to overwrite an existing run. Do not rerun the published
850-scene audit into `/workspace/results/metadata_audit`; that is a shared baseline.

| Path | Use |
| --- | --- |
| datasets/nuscenes | Canonical read-only experiment input |
| checkpoints/published | Maintainer-controlled versioned releases |
| checkpoints/<member>/<run-id> | Member training checkpoints |
| results/<member>/<experiment>/<run-id> | Independent experiment output |
| results/metadata_audit | Existing published audit, preserve unchanged |
| cache/<member> | Rebuildable member cache |
| staging/<member>/<run-id> | Temporary member acquisition/staging |

All table paths are relative to `/workspace`. Do not share a writable venv, Git
checkout, SQLite database, CSV append target or checkpoint filename between Pods.

## Dataset protection: required before routine experiments

Dev is a control-plane role, not a filesystem read-only role. A maintainer must
inspect the actual filesystem permissions and numeric UID/GID mapping first, then:

- Own canonical data with an administrator identity, different from experiment users.
- Allow experiment users to read files and traverse directories, but not write.
- Protect parent directories too, including `/workspace` and `datasets`, so a user
  cannot delete/rename a protected child through a writable parent.
- Give each member a consistent non-root UID/GID across Pods and writable personal
  output directories. Default SSH, notebooks and experiment processes to that user.
- Keep package installation/admin maintenance separate from experiment execution.
  Do not run experiments with sudo or root, or use broad `chmod -R 777` commands.
- Configure acquisition/preprocessing to write to personal staging/results, never
  alongside original samples or metadata. Canonical acquisition is maintainer-only.

Validate on a disposable test fixture on the volume first: read succeeds; create,
overwrite, rename and deletion fail under the experiment identity; personal output
writes succeed. Never test deletion against real dataset files. Then verify actual
dataset ownership/modes and access without modifying data. Recheck after creating
replacement Pods. This protection is not yet applied or tested on the existing volume.

Dev members can still delete the entire volume through the console/API, and root
can defeat ordinary file protections. Never delete, resize or reconfigure shared
storage without explicit team agreement. Do not rely on a shell alias or a Python
path convention as the sole write barrier.

## Rollout acceptance

- [ ] Owner's current account is a Team; existing volume ID/location unchanged.
- [ ] Each invitee accepts and sees that exact volume in Team context.
- [ ] Per-member Cost Centers and shared-storage attribution are configured.
- [ ] Member Pods use EU-RO-1, independent local clones and non-root experiments.
- [ ] Fixture-based permission checks pass; canonical data access is read-only.
- [ ] Member outputs are isolated; published audit files remain untouched.

## Official references

- [Team roles and invitations](https://docs.runpod.io/accounts-billing/manage-accounts)
- [Team resource ownership versus Organizations](https://docs.runpod.io/organizations/orgs-overview)
- [Network volumes](https://docs.runpod.io/storage/network-volumes)
- [Cost Centers](https://docs.runpod.io/accounts-billing/cost-centers)
