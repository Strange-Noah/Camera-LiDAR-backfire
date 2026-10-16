# Storage and recovery

## Layer boundaries

GitHub stores source/configuration, dependency locks, reports, and small validation
outputs. The Network Volume stores datasets, checkpoints, persistent results, cache,
and staging archives. The Pod contains a disposable OS, checkout, and Python venvs.
A checkout on the Pod is never the only copy of code.

Stable layout:

```text
/workspace/
  datasets/nuscenes/
  checkpoints/
  results/
  cache/
  staging/
```

Resource identifiers and the CPU preparation shape are in
`configs/runpod-preaudit.json`. The Network Volume is `p6vb29fmlu` in `EU-RO-1`.
The mount was verified inside the Pod as
`mfs#euro.runpod.net:9421[/networkvolumes/p6vb29fmlu]` with filesystem type `fuse`.
`df` exposes a shared backing filesystem's capacity, not this project's quota.
The Runpod API's allocated **50 GB** is the quota to plan against.

## Grow the dataset later

The official Network Volume documentation allows increasing size, not decreasing it.
The current `update-network-volume` API accepts a new `size` on the existing volume ID.
This is an in-place capacity update; increasing capacity does not require creating a
new volume or downloading the dataset again. No resize has been exercised in this
milestone. Confirm the latest API behavior and available credit before expanding.

Standard storage is documented at $0.07/GB/month for the first 1 TB and
$0.05/GB/month beyond 1 TB. The initial 50 GB is $3.50/month regardless of whether
a Pod is running. Check current prices before making future changes.
Source: https://docs.runpod.io/storage/network-volumes

The datacenter is a placement constraint: a replacement Pod must use the same DC
as this Network Volume. Moving to a different DC requires a separate migration;
volume expansion and cross-DC migration are different operations. CPU availability
and later GPU availability were checked during planning; they are not reservations.

## Replace disposable compute without redownloading

1. Ensure all needed source changes and locks are pushed to GitHub. Save outputs
   under `/workspace/results`, never only under `/root` or `/opt`.
2. Query the existing volume and current compute inventory in `EU-RO-1`.
3. When explicitly retiring an old Pod, terminate only that Pod. Keep Network
   Volume `p6vb29fmlu`; deleting the Network Volume destroys the persistent data.
4. Create the replacement Pod with the existing Network Volume attached at
   `/workspace` **at creation**. Do not create a new empty volume by mistake.
5. Query its current SSH host and port. Verify the host key and connect using a
   registered private key held on the Mac; do not copy private keys into the Pod.
6. Clone GitHub outside the dataset root, install OS prerequisites from README,
   run `prepare_storage.sh`, then `setup_metadata_env.sh` and `verify_nuscenes.sh`.
7. Existing data: skip acquisition entirely. Verification never downloads data.

This procedure is documented but no destructive Pod replacement was performed as
part of the current milestone. The mount and independent storage resource were
verified live; future Pod replacement should be explicitly scheduled.

## Authentication and retention

Runpod MCP uses the user's OAuth authorization. SSH uses the separately registered
public key. nuScenes access and terms were handled by the user; its official mini
URL is documented in the manifest. No credentials belong in Git or the reports.

Network Volume persistence is not a backup guarantee. Storage bills while compute
is idle; maintain enough balance for retention. See Runpod's official documentation
for the consequences of unpaid storage. The original mini archive is retained in
`/workspace/staging`; it is not transferred through the Mac.

## Current milestone boundary

Only mini acquisition and bounded schema/file-access checks are authorized here.
No trainval download, full metadata audit, scene-suitability conclusions, BEVFusion,
MultiCorrupt, checkpoint acquisition, corruption or model inference is performed.
