#!/usr/bin/env python3
"""Run tools/test.py using a frozen camera-offset manifest."""
import argparse
import csv
import json
import os
import sys
import importlib.util
import gc

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from temporal_misalignment import NuScenesCameraIndex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--common-tokens", required=True)
    parser.add_argument("test_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    forwarded = args.test_args[1:] if args.test_args[:1] == ["--"] else args.test_args
    if len(forwarded) < 2:
        parser.error("Pass CONFIG CHECKPOINT and tools/test.py options after --")

    with open(args.common_tokens, encoding="utf-8") as stream:
        common_tokens = set(json.load(stream))
    with open(args.manifest, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    entries = {
        (row["sample_token"], row["channel"]): row
        for row in rows if row["sample_token"] in common_tokens
    }
    index = NuScenesCameraIndex(args.dataroot, args.version)

    from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset

    original_load = NuScenesDataset.load_annotations
    original_get = NuScenesDataset.get_data_info
    original_evaluate = NuScenesDataset.evaluate

    def filtered_load(self, ann_file):
        infos = original_load(self, ann_file)
        return [info for info in infos if info["token"] in common_tokens]

    def temporal_get(self, data_index):
        info = self.data_infos[data_index]
        camera_infos = info["cams"]
        backups = {}
        sample = index.sample_by_token[info["token"]]
        try:
            for channel, camera_info in camera_infos.items():
                key = (info["token"], channel)
                if key not in entries:
                    raise KeyError(f"Manifest is missing {key}")
                row = entries[key]
                selected = index.nusc.get(
                    "sample_data", row["selected_camera_token"]
                )
                original_record = index.nusc.get(
                    "sample_data", row["original_camera_token"]
                )
                observed_error_ms = (
                    (selected["timestamp"] - original_record["timestamp"]) / 1000.0
                    - float(row["requested_offset_ms"])
                )
                if abs(observed_error_ms - float(row["offset_error_ms"])) > 0.01:
                    raise ValueError(f"Manifest timing fields disagree for {key}")
                source_sample = index.nusc.get("sample", selected["sample_token"])
                if source_sample["scene_token"] != sample["scene_token"]:
                    raise ValueError(f"Selected camera crossed scene boundary: {key}")

                backups[channel] = dict(camera_info)
                c2l = index.camera_to_lidar(selected, info)
                calib = index.nusc.get(
                    "calibrated_sensor", selected["calibrated_sensor_token"]
                )
                camera_info["data_path"] = index.nusc.get_sample_data_path(
                    selected["token"]
                )
                camera_info["sample_data_token"] = selected["token"]
                camera_info["sensor2lidar_rotation"] = c2l[:3, :3]
                camera_info["sensor2lidar_translation"] = c2l[:3, 3]
                camera_info["sensor2ego_translation"] = calib["translation"]
                camera_info["sensor2ego_rotation"] = calib["rotation"]
            return original_get(self, data_index)
        finally:
            for channel, old_camera_info in backups.items():
                camera_infos[channel].clear()
                camera_infos[channel].update(old_camera_info)

    def memory_safe_evaluate(self, *eval_args, **eval_kwargs):
        # Inference is complete. Release the timestamp index before the
        # nuScenes evaluator loads its own tables from disk.
        index.by_scene_channel.clear()
        index.times.clear()
        index.sample_by_token.clear()
        index.nusc = None
        gc.collect()
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass
        return original_evaluate(self, *eval_args, **eval_kwargs)

    NuScenesDataset.load_annotations = filtered_load
    NuScenesDataset.get_data_info = temporal_get
    NuScenesDataset.evaluate = memory_safe_evaluate

    # Manifest selection and inference stay in one process for reproducibility.
    if "--cfg-options" not in forwarded:
        forwarded += ["--cfg-options", "data.workers_per_gpu=0"]
    else:
        cfg_pos = forwarded.index("--cfg-options") + 1
        while cfg_pos < len(forwarded) and not forwarded[cfg_pos].startswith("--"):
            cfg_pos += 1
        forwarded.insert(cfg_pos, "data.workers_per_gpu=0")

    sys.argv = [os.path.join(ROOT, "tools", "test.py")] + forwarded
    test_path = sys.argv[0]
    spec = importlib.util.spec_from_file_location("bevfusion_temporal_test", test_path)
    test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_module)
    test_module.main()


if __name__ == "__main__":
    main()


