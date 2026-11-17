"""Utilities for reproducible nuScenes camera/LiDAR timestamp perturbations."""
import csv
import json
import os
from bisect import bisect_left
from collections import defaultdict

import numpy as np
from mmengine.fileio import load
from nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

CAMERA_CHANNELS = (
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
)


class NuScenesCameraIndex:
    """Scene/channel constrained timestamp lookup for camera sample_data."""

    def __init__(self, dataroot, version="v1.0-trainval"):
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        sample_scene = {s["token"]: s["scene_token"] for s in self.nusc.sample}
        self.by_scene_channel = defaultdict(list)
        for record in self.nusc.sample_data:
            channel = record.get("channel")
            if channel not in CAMERA_CHANNELS:
                continue
            scene = sample_scene.get(record["sample_token"])
            if scene is not None:
                self.by_scene_channel[(scene, channel)].append(record)
        self.times = {}
        for key, records in self.by_scene_channel.items():
            records.sort(key=lambda r: r["timestamp"])
            self.times[key] = [r["timestamp"] for r in records]
        self.sample_by_token = {s["token"]: s for s in self.nusc.sample}

    def nearest(self, original_token, channel, offset_ms):
        original = self.nusc.get("sample_data", original_token)
        sample = self.nusc.get("sample", original["sample_token"])
        key = (sample["scene_token"], channel)
        records = self.by_scene_channel[key]
        times = self.times[key]
        target = original["timestamp"] + int(round(offset_ms * 1000.0))
        pos = bisect_left(times, target)
        candidates = [i for i in (pos - 1, pos) if 0 <= i < len(records)]
        selected = min(candidates, key=lambda i: (abs(times[i] - target), times[i]))
        return original, records[selected]

    def camera_to_lidar(self, source_camera_record, lidar_info):
        """Column-vector transform from selected camera coordinates to anchor LiDAR."""
        calib = self.nusc.get(
            "calibrated_sensor", source_camera_record["calibrated_sensor_token"]
        )
        cam_pose = self.nusc.get("ego_pose", source_camera_record["ego_pose_token"])
        camera_to_global = transform_matrix(
            cam_pose["translation"], Quaternion(cam_pose["rotation"])
        ) @ transform_matrix(
            calib["translation"], Quaternion(calib["rotation"])
        )
        lidar_to_global = transform_matrix(
            lidar_info["ego2global_translation"],
            Quaternion(lidar_info["ego2global_rotation"]),
        ) @ transform_matrix(
            lidar_info["lidar2ego_translation"],
            Quaternion(lidar_info["lidar2ego_rotation"]),
        )
        return np.linalg.inv(lidar_to_global) @ camera_to_global


def load_validation_infos(info_path, load_interval=1):
    data = load(info_path)
    infos = sorted(data["infos"], key=lambda item: item["timestamp"])
    return infos[::load_interval]


def make_manifest_rows(index, infos, offsets_ms, tolerance_ms):
    """Make per-offset manifests and a token cohort valid for all conditions."""
    manifests = {float(offset): [] for offset in offsets_ms}
    common_tokens = []
    for info in infos:
        sample = index.sample_by_token[info["token"]]
        lidar = index.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        sample_rows = {float(offset): [] for offset in offsets_ms}
        valid_for_all = True
        for channel in CAMERA_CHANNELS:
            camera_info = info["cams"][channel]
            for offset in offsets_ms:
                original, selected = index.nearest(
                    camera_info["sample_data_token"], channel, float(offset)
                )
                actual_ms = (selected["timestamp"] - original["timestamp"]) / 1000.0
                error_ms = actual_ms - float(offset)
                sample_rows[float(offset)].append({
                    "sample_token": info["token"],
                    "scene_token": sample["scene_token"],
                    "channel": channel,
                    "original_camera_token": original["token"],
                    "selected_camera_token": selected["token"],
                    "lidar_timestamp_us": lidar["timestamp"],
                    "original_camera_timestamp_us": original["timestamp"],
                    "selected_camera_timestamp_us": selected["timestamp"],
                    "clean_camera_lidar_delta_ms":
                        (original["timestamp"] - lidar["timestamp"]) / 1000.0,
                    "requested_offset_ms": float(offset),
                    "actual_offset_ms": actual_ms,
                    "offset_error_ms": error_ms,
                    "actual_camera_lidar_delta_ms":
                        (selected["timestamp"] - lidar["timestamp"]) / 1000.0,
                    "image_path": index.nusc.get_sample_data_path(selected["token"]),
                })
                if abs(error_ms) > tolerance_ms:
                    valid_for_all = False
        if valid_for_all:
            common_tokens.append(info["token"])
            for offset in offsets_ms:
                manifests[float(offset)].extend(sample_rows[float(offset)])
    return manifests, common_tokens


def write_manifest(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_common_tokens(path, tokens):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(tokens, stream, indent=2)
