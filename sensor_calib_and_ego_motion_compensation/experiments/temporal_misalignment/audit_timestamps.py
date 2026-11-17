#!/usr/bin/env python3
"""Print camera frame intervals and clean camera-to-LiDAR timestamp deltas."""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))d
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from temporal_misalignment import CAMERA_CHANNELS, NuScenesCameraIndex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    args = parser.parse_args()
    index = NuScenesCameraIndex(args.dataroot, args.version)
    intervals = defaultdict(list)
    for (scene, channel), records in index.by_scene_channel.items():
        times = [r["timestamp"] for r in records]
        intervals[channel].extend((b - a) / 1000.0 for a, b in zip(times, times[1:]))

    clean_deltas = defaultdict(list)
    for sample in index.nusc.sample:
        lidar = index.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        for channel in CAMERA_CHANNELS:
            camera = index.nusc.get("sample_data", sample["data"][channel])
            clean_deltas[channel].append(
                (camera["timestamp"] - lidar["timestamp"]) / 1000.0
            )

    for channel in CAMERA_CHANNELS:
        frame_dt = np.asarray(intervals[channel])
        clean_dt = np.asarray(clean_deltas[channel])
        print(
            f"{channel}: frame spacing median={np.median(frame_dt):.1f} ms "
            f"(p10={np.percentile(frame_dt, 10):.1f}, "
            f"p90={np.percentile(frame_dt, 90):.1f}); clean cam-LiDAR "
            f"delta median={np.median(clean_dt):.1f} ms "
            f"(p10={np.percentile(clean_dt, 10):.1f}, "
            f"p90={np.percentile(clean_dt, 90):.1f})"
        )


if __name__ == "__main__":
    main()
