from pathlib import Path
import csv
import sys
import os
import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

DATAROOT = "/workspace/nuscenes"
OCAMO_DIR = "/workspace/OCaMo/python/inference"
SCENE_INDEX = int(os.environ.get("SCENE_INDEX", "0"))
INJECT_ROT_DEG = np.fromstring(
    os.environ.get("INJECT_ROT_DEG", "0.5,0,0"),
    sep=",",
)
INJECT_TAG = os.environ.get("INJECT_TAG", "roll_plus0p5")

if INJECT_ROT_DEG.shape != (3,):
    raise ValueError("INJECT_ROT_DEG must contain three comma-separated values")
OUTPUT_DIR = Path(
    "/workspace/bevfusion/results/calibration_baseline"
)
CAMERAS = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
]
NUM_FRAMES = 20
SIGMA_PX = 9.0
K_NEIGHBORS = 10
PERTURBATIONS_DEG = [
    ("zero", [0.0, 0.0, 0.0]),
    ("roll_-0.5", [-0.5, 0.0, 0.0]),
    ("roll_+0.5", [0.5, 0.0, 0.0]),
    ("pitch_-0.5", [0.0, -0.5, 0.0]),
    ("pitch_+0.5", [0.0, 0.5, 0.0]),
    ("yaw_-0.5", [0.0, 0.0, -0.5]),
    ("yaw_+0.5", [0.0, 0.0, 0.5]),
    ("roll_-1.0", [-1.0, 0.0, 0.0]),
    ("roll_+1.0", [1.0, 0.0, 0.0]),
    ("pitch_-1.0", [0.0, -1.0, 0.0]),
    ("pitch_+1.0", [0.0, 1.0, 0.0]),
    ("yaw_-1.0", [0.0, 0.0, -1.0]),
    ("yaw_+1.0", [0.0, 0.0, 1.0]),
]

sys.path.insert(0, OCAMO_DIR)
from include.feature_extraction import get_corners


def make_transform(rotation, translation):
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = Quaternion(rotation).rotation_matrix
    t[:3, 3] = np.asarray(translation, dtype=np.float64)
    return t


def get_ring_ordered_corners(points):
    ring_ids = np.rint(points[:, 4]).astype(np.int32)
    output = []
    for ring_id in np.unique(ring_ids):
        scan = points[ring_ids == ring_id]
        if len(scan) < 11:
            continue
        azimuth = np.arctan2(scan[:, 1], scan[:, 0])
        scan = scan[np.argsort(azimuth)]
        corners = get_corners(scan)[:, :3]
        if len(corners):
            output.append(corners)
    return np.concatenate(output) if output else np.empty((0, 3))


def camera_transform(nusc, lidar_sd, camera_sd, camera_calib):
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
    return np.linalg.inv(camera_to_global) @ lidar_to_global


def score(
    corners, transform, intrinsic, width, height, edge_tree,
    reference_transform=None,
):
    # Candidate transform may include the injected calibration error.
    candidate_base = transform["T_lidar_camera"]

    # Select the fixed corner set using the clean reference transform.
    support_transform = (
        reference_transform
        if reference_transform is not None
        else candidate_base
    )

    # Fix the evaluated corner set using the recorded calibration.
    points_h = np.column_stack([corners, np.ones(len(corners))])
    base_camera = (support_transform @ points_h.T).T[:, :3]
    base_front = base_camera[:, 2] > 0.1
    base_camera = base_camera[base_front]
    base_corners = corners[base_front]

    base_projected = (intrinsic @ base_camera.T).T
    base_pixels = base_projected[:, :2] / base_projected[:, 2:3]
    base_visible = (
        (base_pixels[:, 0] >= 0) & (base_pixels[:, 0] < width)
        & (base_pixels[:, 1] >= height // 3)
        & (base_pixels[:, 1] < height)
    )
    base_corners = base_corners[base_visible]

    if not len(base_corners):
        return float("nan"), 0

    # Apply the candidate rotation to that fixed corner set.
    radians = np.deg2rad(transform["perturb_deg"])
    correction = np.eye(4)
    correction[:3, :3] = Rotation.from_rotvec(radians).as_matrix()
    candidate = candidate_base @ correction

    fixed_h = np.column_stack(
        [base_corners, np.ones(len(base_corners))]
    )
    candidate_camera = (candidate @ fixed_h.T).T[:, :3]
    front = candidate_camera[:, 2] > 0.1

    projected = (intrinsic @ candidate_camera.T).T
    pixels = np.full((len(base_corners), 2), np.nan)
    pixels[front] = projected[front, :2] / projected[front, 2:3]
    rounded = np.zeros((len(base_corners), 2), dtype=np.int32)
    rounded[front] = np.rint(pixels[front]).astype(np.int32)

    inside = (
        front
        & (rounded[:, 0] >= 0) & (rounded[:, 0] < width)
        & (rounded[:, 1] >= height // 3)
        & (rounded[:, 1] < height)
    )

    # An out-of-view corner contributes zero instead of disappearing.
    similarities = np.zeros(len(base_corners), dtype=np.float64)
    if inside.any():
        k = min(K_NEIGHBORS, len(edge_tree.data))
        distances, _ = edge_tree.query(rounded[inside], k=k)
        similarities[inside] = np.exp(
            -(distances ** 2) / (2 * SIGMA_PX ** 2)
        ).mean(axis=1)

    return float(similarities.mean()), int(inside.sum())


nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=False)
scene = nusc.scene[SCENE_INDEX]
OUTPUT = OUTPUT_DIR / f"rotation_recovery_{INJECT_TAG}_{scene['name']}.csv"
sample_token = scene["first_sample_token"]
rows = []

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
        camera_file = Path(DATAROOT) / camera_sd["filename"]
        image = cv2.imread(str(camera_file))
        if image is None:
            continue

        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 100)
        edges[: height // 3] = 0
        edge_y, edge_x = np.where(edges > 0)
        if not len(edge_x):
            continue

        edge_tree = KDTree(np.column_stack([edge_x, edge_y]))
        calib = nusc.get(
            "calibrated_sensor", camera_sd["calibrated_sensor_token"]
        )
        # Save the unmodified transform for the fixed corner support.
        clean_transform = camera_transform(
            nusc, lidar_sd, camera_sd, calib
        )
        # Simulate a known +0.5 degree rotation error in the script's
        # LiDAR-coordinate rotation parameterization.
        injection = np.eye(4)
        injection[:3, :3] = Rotation.from_rotvec(
            np.deg2rad(INJECT_ROT_DEG)
        ).as_matrix()
        base_transform = clean_transform @ injection
        intrinsic = np.asarray(calib["camera_intrinsic"], dtype=np.float64)

        frame_results = []
        for name, perturb in PERTURBATIONS_DEG:
            holder = {
                "T_lidar_camera": base_transform,
                "perturb_deg": perturb,
            }
            value, visible = score(
                                corners,
                                holder,
                                intrinsic,
                                width,
                                height,
                                edge_tree,
                                reference_transform=clean_transform,
                            )
            rows.append({
                "frame": frame_i,
                "sample_token": sample["token"],
                "channel": channel,
                "perturbation": name,
                "roll_deg": perturb[0],
                "pitch_deg": perturb[1],
                "yaw_deg": perturb[2],
                "score": value,
                "projected_corners": visible,
            })
            frame_results.append((name, value, visible))

    sample_token = sample["next"]

with OUTPUT.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("Scene:", scene["name"])
print("Wrote:", OUTPUT)
print("Higher score means closer projected-corner / image-edge agreement.")
