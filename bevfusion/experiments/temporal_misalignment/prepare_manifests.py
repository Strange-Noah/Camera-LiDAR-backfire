#!/usr/bin/env python3
"""Prepare timestamp-based camera offset manifests on a shared sample subset."""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from temporal_misalignment import (
    CAMERA_CHANNELS,
    NuScenesCameraIndex,
    load_validation_infos,
    make_manifest_rows,
    write_common_tokens,
    write_manifest,
)


def label(offset):
    if float(offset) == 0:
        return "zero"
    return ("plus" if offset > 0 else "minus") + f"{abs(offset):g}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--info", required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--offsets-ms", nargs="+", type=float,
                        default=[-500, -250, -100, 0, 100, 250, 500])
    parser.add_argument("--tolerance-ms", type=float, default=50.0)
    parser.add_argument("--load-interval", type=int, default=1)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if args.load_interval < 1 or args.tolerance_ms < 0:
        parser.error("load-interval must be positive and tolerance nonnegative")
    if 0.0 not in args.offsets_ms:
        parser.error("Include 0 in offsets-ms as the clean condition")

    index = NuScenesCameraIndex(args.dataroot, args.version)
    infos = load_validation_infos(args.info, args.load_interval)
    manifests, common_tokens = make_manifest_rows(
        index, infos, args.offsets_ms, args.tolerance_ms
    )
    os.makedirs(args.out_dir, exist_ok=True)
    for offset, rows in manifests.items():
        suffix = label(offset)
        write_manifest(os.path.join(args.out_dir, f"manifest_{suffix}ms.csv"), rows)
        actual = np.asarray([float(row["actual_offset_ms"]) for row in rows])
        if len(actual):
            count = len(rows) // len(CAMERA_CHANNELS)
            print(
                f"{suffix}ms: {count} shared samples; actual perturbation "
                f"median={np.median(actual):.1f} ms; "
                f"range={actual.min():.1f}..{actual.max():.1f} ms"
            )
        else:
            print(f"{suffix}ms: no shared samples")
    token_path = os.path.join(args.out_dir, "common_sample_tokens.json")
    write_common_tokens(token_path, common_tokens)
    summary = {
        "requested_offsets_ms": args.offsets_ms,
        "tolerance_ms": args.tolerance_ms,
        "load_interval": args.load_interval,
        "input_samples": len(infos),
        "common_samples": len(common_tokens),
        "common_tokens_file": token_path,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
