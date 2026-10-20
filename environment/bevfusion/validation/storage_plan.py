"""Build a capacity plan from completed archive measurements, never tgz ratios."""
import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--existing-bytes', type=int, required=True,
                        help='Fresh du -sx --block-size=1 /workspace after CPU preparation')
    args = parser.parse_args()
    measurements = json.loads((args.evidence/'storage-measurements.json').read_text())
    assert measurements['status'] == 'passed'
    assert measurements['sample_count'] == 6019
    rows = []
    detail = {}
    for scenario, values in measurements['scenarios'].items():
        # Acquisition reads gzip streams without saving archives. Allow for two
        # active member .part files, abandoned partials and atomic JSON rewrites.
        temporary = 1_000_000_000
        outputs = 3_000_000_000
        subtotal = args.existing_bytes + values['additional_payload_bytes'] + temporary + outputs
        margin = max(10_000_000_000, math.ceil(.15 * subtotal))
        required = subtotal + margin
        capacity = max(50, math.ceil(required/10_000_000_000)*10)
        incremental = max(0, capacity-50)*.07
        detail[scenario] = dict(existing_bytes=args.existing_bytes, **values,
            temporary_allowance_bytes=temporary, output_allowance_bytes=outputs,
            margin_bytes=margin, required_bytes=required, recommended_volume_gb=capacity,
            standard_storage_usd_per_gb_month=.07, incremental_usd_per_month=round(incremental,2))
        rows.append('| %s | %s | %.3f | %.3f | %.3f | %s | $%.2f |' % (scenario,
            format(values['files'],','),values['payload_bytes']/1e9,values['already_available_payload_bytes']/1e9,
            values['additional_payload_bytes']/1e9, capacity, incremental))
    maps_path = args.evidence/'map-acquisition-report.json'
    maps = json.loads(maps_path.read_text()) if maps_path.exists() else None
    text = '''# Validation storage plan

This plan uses actual uncompressed tar-member sizes for all official validation
sensor files, with globally deduplicated paths. All ten official trainval blob
archives were inspected; gzip CRC/footer and HTTP byte counts were checked.
No capacity is extrapolated from compressed archive sizes.

Dataset: v1.0-trainval, official val, 150 scenes, 6019 samples.
Network Volume: p6vb29fmlu, EU-RO-1, STANDARD, currently 50 GB. No resize performed.
GB below means 10^9 bytes; GiB = bytes / 2^30. Provision conservatively against
decimal GB; do not confuse a filesystem's displayed TiB with purchased quota.

| Scenario | Sensor files | Gross sensor GB | Already available GB | Additional sensor GB | Recommended capacity GB | Incremental USD/month |
|---|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

A: six keyframe cameras + current LiDAR + the union of every possible previous
sweep from the pinned actual test pipeline. The official converter stores ten
previous sweeps, while the loader randomly chooses nine when ten exist; its
test_mode parameter is left at the official default False. Deduplication is
global, including overlap between current frames and other samples' histories.

B: A plus all recorded non-keyframe cameras in the same validation scenes.
This is an explicitly declared preparation superset, not a chosen corruption
protocol. No temporal offsets have been implemented. Non-keyframe availability
was verified in actual official archive headers. Some future offsets may need
fewer frames; this plan does not claim B is the smallest such experiment.

## Accounting

The fresh existing-volume measurement is **%s bytes** (%.3f GB / %.3f GiB).
It includes existing mini files, full metadata, runtime/checkpoints, acquired
current keyframes, derived metadata/infos, maps if acquired, and existing logs.

To avoid counting existing validation files twice, use:

`required = existing + gross_val - already_available_val + temporary + future_outputs + margin`

Equivalently, the requested `existing + val + temporary + results + margin`
formula uses **incremental missing validation payload** as `val`. Gross bytes
and overlap are reported separately above. Existing metadata/calibration/ego
poses/annotations are retained, not counted again as new sensor payload.

Temporary allowance: 1 GB for concurrent .part files, interrupted extraction and
atomic manifest/config rewrites. Sensor tgz archives are streamed, never retained.
Map zip temporary storage uses disposable CPU container disk and is not retained
on the Network Volume. Future prediction/evaluation/log allowance: 3 GB for both
models, conservatively above extrapolation from the existing smoke artifacts.
This is a budget allowance, not a measured full-val prediction size. Existing
infos already occupy space in `existing`; the 3 GB remains additional headroom.

Margin: max(10 GB, 15%% of existing + missing payload + temporary + outputs).
Capacity is rounded upward to a 10 GB step. Costs use STANDARD under-1-TB pricing
of $0.07/GB/month, prorated by the provider; no GPU/CPU compute is included.
Increment = (recommended capacity - 50) * $0.07. Existing 50 GB nominal cost is
$3.50/month. Prices are reference rates, not an account invoice or tax estimate.
Source: [Runpod storage pricing](https://www.runpod.io/pricing).

## Remaining checks before acquisition/GPU

Resize requires the user's explicit authorization after reviewing this plan.
After approval, refresh du/quota, re-inventory existing files, acquire only the
missing A members, verify all sizes/hashes/JPEG/LiDAR structures on CPU, and run
full official-pipeline dataset iteration. Full GPU inference requires a separate
authorization after every CPU admission gate passes. Neither approval is implied
by this capacity calculation.
''' % (format(args.existing_bytes,','),args.existing_bytes/1e9,args.existing_bytes/2**30)
    if maps:
        text+='\nOfficial map JSON members retained: **%s bytes**; source zip %s bytes. These satisfy the original LoadBEVSegmentation stage without changing sensor preprocessing. They are already included in the fresh existing-volume measurement.\n' % (format(maps['retained_uncompressed_bytes'],','),format(maps['compressed_bytes'],','))
    (args.evidence/'validation-storage-plan.md').write_text(text)
    (args.evidence/'capacity-calculation.json').write_text(json.dumps(detail,indent=2)+'\n')
    print(json.dumps(detail,indent=2))


if __name__ == '__main__':
    main()
