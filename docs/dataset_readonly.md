# Dataset read-only policy

## Scope

The shared canonical dataset is `/workspace/datasets/nuscenes` on Network Volume
`p6vb29fmlu` in EU-RO-1. The policy protects non-root experiment processes against
accidental writes, not against root maintenance or Runpod Dev control-plane deletion.
Team account conversion was confirmed by the owner; invitations and Cost Centers
are separate from filesystem permissions.

## Persistent filesystem policy

- Every dataset file: root:root, `0444`.
- Every dataset directory: root:root, `0555`.
- `/workspace` and `/workspace/datasets`: root:root, `0555`, protecting child names on the observed FUSE backend.
- Member output directories: owned by the member's numeric UID/GID, `0750`.

Do not place caches or generated metadata in the dataset tree. Code that tries to
write beside input files must instead use personal cache/staging/results paths.
The standard audit and smoke-test shell entrypoints refuse root execution when
the protection report exists and launch experiments through Landlock. The provisioned
member SSH login shell also automatically enters Landlock, inherited by its children.
No unprotected fallback is allowed if the host lacks Landlock ABI >= 3. The tested
host supports ABI 4. Write access is limited to the local home, /tmp, /dev/shm and
the four per-member volume directories, plus /dev/null and /dev/tty. Reads remain
allowed. See [Linux kernel documentation](https://docs.kernel.org/userspace-api/landlock.html).

The mount remains read-write for administrator maintenance and result writes;
this uses file modes plus a Linux Landlock process write allowlist, not a Runpod
read-only mount flag. The observed FUSE backend allowed non-root chmod and rename
through 0755 parents; mode bits alone are insufficient. Landlock blocks content
writes even after chmod, but does not protect chmod metadata changes on ABI 4.

## Provision every new or restarted Pod

A Pod stop can discard its local users/home/environment. Dataset modes persist on
the volume, but **each replacement or restarted Pod needs its non-root login again**.
A default Runpod root shell, web terminal or root Jupyter session is NOT protected.

From an administrator checkout on the Pod:

```bash
# One-time dataset maintenance only, not required on every new Pod:
python3 scripts/protect_dataset.py --apply

# Per-Pod setup; administrator assigns a stable unique UID to each member.
# 20001 / researcher is reserved for the protection smoke test.
# Replace member-a and 20010 with the agreed member identity.
bash scripts/setup_experiment_user.sh member-a 20010 /path/to/member-authorized-keys
python3 scripts/verify_dataset_permissions.py --user member-a
```

The authorized-keys file contains PUBLIC keys only. No password, API key or private
key belongs in Git or on the shared volume. Record numeric member IDs consistently
across Pods; do not reuse another member's UID. Run the member's SSH session directly
as that username using the Pod's direct SSH host/port, not as root. The Runpod proxy
root session is for maintenance only. Do not grant passwordless sudo or run notebooks
as root for experiments. Keep root as the deliberate administrator entrypoint.

The setup script installs a protected login shell under /usr/local/bin and the
launcher under /usr/local/lib/camera-lidar. It refuses an occupied UID, unexpected supplementary groups or an
existing output directory owned by someone else. It creates personal results,
checkpoints, cache and staging directories. It does not install a Python stack.
After login, clone into your local home and use `configs/member.env.example` with
MEMBER_SLUG matching this Linux login. Never share one Git working tree or venv.

## Verification and recovery evidence

`/workspace/results/storage_permissions/` stores:

- `<timestamp>-before.jsonl.gz`: original per-path UID/GID/mode/size/mtime; root-only.
- `protection_report.json`: complete permissions check and unchanged content hashes
  for all metadata/provenance JSON files plus one camera and one LiDAR file.
- `access_verification.json`: operations executed as the non-root experiment user.

The verifier reads actual mini/trainval metadata and camera/LiDAR files. It tests
create, overwrite, delete, rename and chmod against a disposable fixture only,
including rename through its parent and a content-write attempt after chmod. It also verifies writable personal output
paths. No deletion probe touches real dataset files. All real file sizes and mtimes
are checked by the protection script; hashes are not a complete sensor-data hash audit.

The permission snapshot permits reviewed restoration of original modes if needed;
there is intentionally no automatic recursive rollback. For deliberate canonical
data updates, use a maintainer root session, record provenance, then reapply and
verify protection. Read-only mode does not replace external backups or protect
against a Dev member deleting the entire Network Volume via Runpod.

## Live acceptance

The maintenance Pod passed real non-root direct SSH login and fixture tests.
See `results/storage_permissions/` for the saved deployment evidence. The completion
report is authoritative for the executed checks; per-member Pod provisioning remains
required for every teammate and every replacement Pod.

GPU workloads and externally started notebook services are not validated here.
Start notebooks inside the protected member shell; a root notebook started by a
template is outside the protection. SFTP subsystems may not use the login shell;
do not treat them as a protected experiment entrypoint. Root, separate unguarded
processes, remote APIs and Team Dev volume deletion remain outside this boundary.
