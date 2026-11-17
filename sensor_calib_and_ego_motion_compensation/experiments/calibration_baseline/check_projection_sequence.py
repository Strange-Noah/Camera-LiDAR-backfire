from pathlib import Path
import sys

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

DATAROOT = "/workspace/nuscenes"
OCAMO_DIR = "/workspace/OCaMo/python/inference"
OUT = Path("/workspace/bevfusion/results/calibration_baseline/projection_sequence")
CAMERAS = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
]
NUM_FRAMES = 20

sys.path.insert(0, OCAMO_DIR)
from include.feature_extraction import get_corners


def make_transform(rotation, translation):
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = Quaternion(rotation).rotation_matrix
    t[:3, 3] = np.asarray(translation, dtype=np.float64)
    return t

def get_ring_ordered_corners(points):
    ring_ids = np.rint(points[:, 4]).astype(np.int32)
    ring_corners = []

    for ring_id in np.unique(ring_ids):
        scan = points[ring_ids == ring_id]
        if len(scan) < 11:
            continue

        azimuth = np.arctan2(scan[:, 1], scan[:, 0])
        scan = scan[np.argsort(azimuth)]

        corners = get_corners(scan)[:, :3]
        if len(corners):
            ring_corners.append(corners)

    if not ring_corners:
        return np.empty((0, 3), dtype=np.float32)

    return np.concatenate(ring_corners, axis=0)


def project(points_lidar, lidar_sd, camera_sd, camera_calib, nusc):
    lidar_calib = nusc.get(
        "calibrated_sensor", lidar_sd["calibrated_sensor_token"]
    )
    lidar_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    camera_pose = nusc.get("ego_pose", camera_sd["ego_pose_token"])

    lidar_to_global = (
        make_transform(lidar_pose["rotation"], lidar_pose["translation"])
        @ make_transform(lidar_calib["rotation"], lidar_calib["translation"])
    )
    camera_to_global = (
        make_transform(camera_pose["rotation"], camera_pose["translation"])
        @ make_transform(camera_calib["rotation"], camera_calib["translation"])
    )
    lidar_to_camera = np.linalg.inv(camera_to_global) @ lidar_to_global

    homogeneous = np.column_stack(
        [points_lidar, np.ones(len(points_lidar), dtype=np.float64)]
    )
    points_camera = (lidar_to_camera @ homogeneous.T).T[:, :3]

    intrinsic = np.asarray(camera_calib["camera_intrinsic"], dtype=np.float64)
    in_front = points_camera[:, 2] > 0.1
    projected = (intrinsic @ points_camera[in_front].T).T
    pixels = projected[:, :2] / projected[:, 2:3]
    return pixels


nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=False)
OUT.mkdir(parents=True, exist_ok=True)

# Follow consecutive samples in one scene.
scene = nusc.scene[0]
sample_token = scene["first_sample_token"]
metric_rows = []

for frame_i in range(NUM_FRAMES):
    if not sample_token:
        break

    sample = nusc.get("sample", sample_token)
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    lidar_file = Path(DATAROOT) / lidar_sd["filename"]
    points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 5)
    corners = get_ring_ordered_corners(points)

    for channel in CAMERAS:
        camera_token = sample["data"].get(channel)
        if not camera_token:
            continue

        camera_sd = nusc.get("sample_data", camera_token)
        camera_calib = nusc.get(
            "calibrated_sensor", camera_sd["calibrated_sensor_token"]
        )
        image_path = Path(DATAROOT) / camera_sd["filename"]
        image = cv2.imread(str(image_path))
        if image is None:
            print("SKIP unreadable image:", image_path)
            continue

        h, w = image.shape[:2]
        pixels = project(corners, lidar_sd, camera_sd, camera_calib, nusc)
        inside = (
            (pixels[:, 0] >= 0) & (pixels[:, 0] < w)
            & (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
        )
        xy = np.round(pixels[inside]).astype(np.int32)
        # Rounding can move an in-bounds coordinate onto the image boundary.
        xy = xy[
            (xy[:, 0] >= 0) & (xy[:, 0] < w)
            & (xy[:, 1] >= 0) & (xy[:, 1] < h)
        ]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 100)
        edges[: h // 3] = 0
        distance = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)

        if len(xy):
            d = distance[xy[:, 1], xy[:, 0]]
            median = float(np.median(d))
            within3 = float(np.mean(d <= 3))
            within5 = float(np.mean(d <= 5))
            for x, y in xy:
                cv2.circle(image, (int(x), int(y)), 2, (0, 0, 255), -1)
        else:
            median = within3 = within5 = float("nan")

        delta_ms = (camera_sd["timestamp"] - lidar_sd["timestamp"]) / 1000.0
        metric_rows.append((channel, median, within3, within5, delta_ms))

        out_file = OUT / f"{frame_i:02d}_{channel}.jpg"
        cv2.imwrite(str(out_file), image)
        print(
            f"frame={frame_i:02d} {channel}: "
            f"corners_in_image={len(xy)} median_edge_px={median:.2f} "
            f"within3={within3:.1%} within5={within5:.1%} "
            f"camera_minus_lidar={delta_ms:.1f}ms"
        )

    sample_token = sample["next"]

print(f"\nScene: {scene['name']}")
print(f"Saved overlays to: {OUT}")
