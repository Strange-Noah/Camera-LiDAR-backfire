from pathlib import Path
import csv
import importlib
import os
import sys
import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

DATAROOT = "/workspace/nuscenes"
OCAMO_DIR = "/workspace/OCaMo/python/inference"
SCENE_INDEX = int(os.environ.get("SCENE_INDEX", "0"))
INJECTED_ROLL_DEG = float(os.environ.get("INJECTED_ROLL_DEG", "0.5"))
OBJECTIVE_MODE = os.environ.get("OBJECTIVE_MODE", "kernel_sum")
MAX_EDGE_DIST_PX = 20.0
sys.path.insert(0, OCAMO_DIR)
from include.feature_extraction import get_corners
# Patch only the imported OCaMo optimizer's objective function.
sgd_module = importlib.import_module("include.SGDSchaul")

current_frame = {}

def make_transform(rotation, translation):
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = Quaternion(rotation).rotation_matrix
    t[:3, 3] = np.asarray(translation, dtype=np.float64)
    return t


def ring_ordered_corners(points):
    """Run the OCaMo corner extractor separately on each ordered LiDAR ring."""
    # Invalid coordinates or ring values must not reach the corner extractor.
    points = points[np.isfinite(points[:, :5]).all(axis=1)]
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float64)

    ring_ids = np.rint(points[:, 4]).astype(np.int32)
    all_corners = []

    for ring_id in np.unique(ring_ids):
        scan = points[ring_ids == ring_id]
        if len(scan) < 11:
            continue
        azimuth = np.arctan2(scan[:, 1], scan[:, 0])
        scan = scan[np.argsort(azimuth)]
        with np.errstate(divide="ignore", invalid="ignore"):
            corners = get_corners(scan)[:, :3]
        corners = corners[np.isfinite(corners).all(axis=1)]
        if len(corners):
            all_corners.append(corners)

    if not all_corners:
        return np.empty((0, 3), dtype=np.float64)
    return np.concatenate(all_corners, axis=0)


def lidar_to_camera_transform(nusc, lidar_sd, camera_sd, camera_calib):
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


def select_initial_support(corners, bad_transform, intrinsic, width, height):
    """Freeze corners visible under the transform available to the tracker."""
    if not len(corners):
        return corners

    points_camera = (
        bad_transform[:3, :3] @ corners.T
        + bad_transform[:3, 3:4]
    ).T
    in_front = points_camera[:, 2] > 0.1
    points_camera = points_camera[in_front]
    front_corners = corners[in_front]

    projected = (intrinsic @ points_camera.T).T
    pixels = projected[:, :2] / projected[:, 2:3]
    visible = (
        (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
        & (pixels[:, 1] >= height / 3) & (pixels[:, 1] < height)
    )
    return front_corners[visible]


def nuscenes_kernel_correlation(kdtree, pcl, params, sigma, k):
    """Objective minimized by the tracker; fixed_scores stores its negative."""
    if len(pcl) == 0 or len(kdtree.data) == 0:
        return 0.0

    transform = current_frame["bad_transform"]
    intrinsic = current_frame["intrinsic"]
    width = current_frame["width"]
    height = current_frame["height"]

    correction = Rotation.from_rotvec(np.asarray(params[:3])).as_matrix()
    points_lidar = correction @ pcl[:, :3].T
    points_camera = (
        transform[:3, :3] @ points_lidar
        + transform[:3, 3:4]
    )

    in_front = points_camera[2] > 0.1
    similarity = np.zeros(len(pcl), dtype=np.float64)
    edge_distance = np.full(
        len(pcl), MAX_EDGE_DIST_PX, dtype=np.float64
    )

    if in_front.any():
        camera_points = points_camera[:, in_front]
        projected = intrinsic @ camera_points
        pixels = projected[:2] / projected[2:3]

        inside = (
            (pixels[0] >= 0) & (pixels[0] < width)
            & (pixels[1] >= height / 3) & (pixels[1] < height)
        )

        if inside.any():
            xy = pixels[:, inside].T
            front_indices = np.flatnonzero(in_front)
            visible_indices = front_indices[inside]

            # Robust Chamfer variant: nearest edge distance, capped at 20 px.
            nearest, _ = kdtree.query(xy, k=1)
            edge_distance[visible_indices] = np.minimum(
                nearest, MAX_EDGE_DIST_PX
            )

            # Existing kernel similarity using up to k nearby edges.
            neighbor_count = min(k, len(kdtree.data))
            distances, _ = kdtree.query(xy, k=neighbor_count)
            if distances.ndim == 1:
                distances = distances[:, None]

            local_similarity = np.exp(
                -(distances ** 2) / (2 * sigma ** 2)
            ).sum(axis=1)
            similarity[visible_indices] = local_similarity

    if OBJECTIVE_MODE == "kernel_sum":
        return -float(similarity.sum())

    if OBJECTIVE_MODE == "kernel_mean":
        return -float(similarity.mean())

    if OBJECTIVE_MODE == "robust_chamfer":
        return float(edge_distance.mean())

    raise ValueError(f"Unknown OBJECTIVE_MODE: {OBJECTIVE_MODE}")

# The original SGDSchaul class resolves this function from its module globals.
sgd_module.estimate_kernel_correlation = nuscenes_kernel_correlation

nusc = NuScenes(
    version="v1.0-trainval",
    dataroot=DATAROOT,
    verbose=False,
)
scene = nusc.scene[SCENE_INDEX]
injection_tag = (
    f"{INJECTED_ROLL_DEG:+.2f}"
    .replace("+", "plus")
    .replace("-", "minus")
    .replace(".", "p")
)
output_path = Path(
    f"/workspace/bevfusion/results/calibration_baseline/"
    f"online_tracker_roll_{OBJECTIVE_MODE}_inj_{injection_tag}_"
    f"runtime_support_{scene['name']}.csv"
)
output_path.parent.mkdir(parents=True, exist_ok=True)

# The synthetic injection and clean transform are ground-truth references for
# this diagnostic only. The tracker below receives only the perturbed
# starting transform and sensor-derived support.
injection = np.eye(4)
injection[:3, :3] = Rotation.from_rotvec(
    np.deg2rad([INJECTED_ROLL_DEG, 0.0, 0.0])
).as_matrix()

# Track only the roll-like x component. The expected correction is -0.5°.
tracker = sgd_module.SGDSchaul(
    params_trck=np.array([0]),
    theta_init=np.zeros(3),
    delta=np.array([0.001, 0.001, 0.001]),
    bnd=np.array([0.0166, 0.0083, 0.0026]),
)

rows = []
sample_token = scene["first_sample_token"]
frame_index = 0

while sample_token:
    sample = nusc.get("sample", sample_token)
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    camera_sd = nusc.get("sample_data", sample["data"]["CAM_FRONT"])

    lidar_path = Path(DATAROOT) / lidar_sd["filename"]
    camera_path = Path(DATAROOT) / camera_sd["filename"]

    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
    corners = ring_ordered_corners(points)

    image = cv2.imread(str(camera_path))
    if image is None:
        print("Skipping unreadable image:", camera_path)
        sample_token = sample["next"]
        frame_index += 1
        continue

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 100)
    edges[: height // 3] = 0
    edge_y, edge_x = np.where(edges > 0)

    if len(edge_x) == 0 or len(corners) == 0:
        print(f"Skipping frame {frame_index}: no edges or corners")
        sample_token = sample["next"]
        frame_index += 1
        continue

    edge_tree = KDTree(np.column_stack([edge_x, edge_y]))
    camera_calib = nusc.get(
        "calibrated_sensor", camera_sd["calibrated_sensor_token"]
    )
    intrinsic = np.asarray(
        camera_calib["camera_intrinsic"], dtype=np.float64
    )

    clean_transform = lidar_to_camera_transform(
        nusc, lidar_sd, camera_sd, camera_calib
    )
    bad_transform = clean_transform @ injection
    if frame_index == 0 and len(corners):
        inverse_correction = Rotation.from_rotvec(
            np.deg2rad([-INJECTED_ROLL_DEG, 0.0, 0.0])
        ).as_matrix()

        clean_xyz = (
            clean_transform[:3, :3] @ corners.T
            + clean_transform[:3, 3:4]
        )
        recovered_xyz = (
            bad_transform[:3, :3] @ inverse_correction @ corners.T
            + bad_transform[:3, 3:4]
        )

        max_difference_m = np.max(np.abs(clean_xyz - recovered_xyz))
        print(f"Inverse-transform check: max difference={max_difference_m:.3e} m")
    # Freeze support selected from the available starting calibration. The
    # clean transform and GT annotations are not used by the estimator.
    support = select_initial_support(
        corners, bad_transform, intrinsic, width, height
    )

    current_frame.update({
        "bad_transform": bad_transform,
        "intrinsic": intrinsic,
        "width": width,
        "height": height,
    })
    
    fixed_corrections_deg = [
        round(i * 0.1, 1) for i in range(-8, 9)
    ]

    fixed_scores = {}
    for angle_deg in fixed_corrections_deg:
        params = np.deg2rad([angle_deg, 0.0, 0.0])
        score = -nuscenes_kernel_correlation(edge_tree, support, params, 9, 10)
        fixed_scores[f"score_roll_{angle_deg:+.1f}"] = score

    if len(support) > 0:
        tracker.update(support, edge_tree)

    estimated_deg = float(np.rad2deg(tracker.theta_[0]))
    rows.append({
        "frame": frame_index,
        "sample_token": sample["token"],
        "scene_name": scene["name"],
        "camera": "CAM_FRONT",
        "injected_roll_deg": INJECTED_ROLL_DEG,
        "expected_correction_deg": -INJECTED_ROLL_DEG,
        "estimated_roll_deg": estimated_deg,
        "absolute_error_deg": abs(estimated_deg + INJECTED_ROLL_DEG),
        "support_corners": len(support),
        "camera_minus_lidar_ms":
            (camera_sd["timestamp"] - lidar_sd["timestamp"]) / 1000.0,
        **fixed_scores,
    })

    if frame_index % 5 == 0:
        print(
            f"frame={frame_index:02d} "
            f"estimate={estimated_deg:+.3f} deg "
            f"expected={-INJECTED_ROLL_DEG:+.3f} deg"
            f" support={len(support)}"
        )

    sample_token = sample["next"]
    frame_index += 1

with output_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("Scene:", scene["name"])
print("Frames:", len(rows))
print("Saved:", output_path)
if rows:
    print("Final estimate:", f"{rows[-1]['estimated_roll_deg']:+.3f} deg")
    print("Expected correction:", f"{-INJECTED_ROLL_DEG:+.3f} deg")
