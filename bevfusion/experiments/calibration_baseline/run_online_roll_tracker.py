from pathlib import Path
import csv
import importlib
import os
import sys
from mmengine.fileio import load
import cv2
import numpy as np
import gc
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

DATAROOT = "/workspace/nuscenes"
OCAMO_DIR = "/workspace/OCaMo/python/inference"
SCENE_INDEX = int(os.environ.get("SCENE_INDEX", "0"))
INJECTED_ROLL_DEG = float(os.environ.get("INJECTED_ROLL_DEG", "0.5"))
OBJECTIVE_MODE = os.environ.get("OBJECTIVE_MODE", "kernel_sum")
SUPPORT_MODE = os.environ.get("SUPPORT_MODE", "clean").lower()
USE_GT_MOVING_FILTER = os.environ.get("USE_GT_MOVING_FILTER", "1") == "1"
MATCHED_SEED = int(os.environ.get("MATCHED_SEED", "0"))
SPLIT_SUPPORT_CHECK = os.environ.get("SPLIT_SUPPORT_CHECK", "0") == "1"
SPLIT_PEAK_AGREEMENT_DEG = 0.10
CAMERA_CHANNEL = os.environ.get("CAMERA_CHANNEL", "CAM_FRONT")
CAMERA_CHANNELS = (
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
)
if CAMERA_CHANNEL not in CAMERA_CHANNELS:
    raise ValueError(
        f"CAMERA_CHANNEL must be one of {CAMERA_CHANNELS}; got {CAMERA_CHANNEL!r}"
    )
if SUPPORT_MODE not in {
    "clean", "runtime", "runtime_matched", "search_stable", "shared_diagnostic"
}:
    raise ValueError(
        "SUPPORT_MODE must be 'clean', 'runtime', 'runtime_matched', "
        "'search_stable', or 'shared_diagnostic'"
    )
MAX_EDGE_DIST_PX = 20.0
ORIENTATION_SIGMA_DEG = 25.0
# Keep these bounds synchronized with SGDSchaul's bnd below. The first
# component is the roll search bound; the other axes are not tracked here.
TRACKER_BOUNDS_RAD = np.array([0.0166, 0.0083, 0.0026], dtype=np.float64)
SUPPORT_MAX_STEP_DEG = 0.05
DIAGNOSTIC_INJECTIONS_DEG = (
    -0.75, -0.50, -0.25, 0.0, 0.25, 0.50, 0.75
)
sys.path.insert(0, OCAMO_DIR)
from include.feature_extraction import get_corners
INFO_PATHS = [
    Path("/workspace/nuscenes/nuscenes_infos_train.pkl"),
    Path("/workspace/nuscenes/nuscenes_infos_val.pkl"),
]

info_by_token = {}
if USE_GT_MOVING_FILTER:
    for info_path in INFO_PATHS:
        if not info_path.exists():
            print(f"Info file not found, skipping: {info_path}")
            continue

        payload = load(str(info_path))
        records = payload["infos"] if isinstance(payload, dict) else payload

        for record in records:
            info_by_token[record["token"]] = {
                "gt_boxes": record["gt_boxes"],
                "gt_velocity": record["gt_velocity"],
                "gt_names": record["gt_names"],
            }

        del records, payload
        gc.collect()

    if not info_by_token:
        raise FileNotFoundError(
            "GT filtering is enabled but no nuScenes info records loaded. "
            "Check INFO_PATHS or set USE_GT_MOVING_FILTER=0."
        )

# Patch only the imported OCaMo optimizer's objective function.
sgd_module = importlib.import_module("include.SGDSchaul")

current_frame = {}

def remove_moving_object_corners(corners, info, speed_threshold=0.5, margin=0.3):
    boxes = np.asarray(info["gt_boxes"])
    velocities = np.asarray(info["gt_velocity"])
    names = np.asarray(info["gt_names"])

    moving_classes = {
        "car", "truck", "bus", "trailer", "construction_vehicle",
        "pedestrian", "motorcycle", "bicycle",
    }
    keep = np.ones(len(corners), dtype=bool)

    for box, velocity, name in zip(boxes, velocities, names):
        speed = np.linalg.norm(velocity)
        if name not in moving_classes or not np.isfinite(speed) or speed < speed_threshold:
            continue

        cx, cy, cz, width, length, height, yaw = box[:7]
        dx = corners[:, 0] - cx
        dy = corners[:, 1] - cy
        dz = corners[:, 2] - cz

        # Rotate LiDAR-frame offsets into the box's local frame.
        local_x = np.cos(yaw) * dx + np.sin(yaw) * dy
        local_y = -np.sin(yaw) * dx + np.cos(yaw) * dy

        inside = (
            (np.abs(local_x) <= width / 2 + margin)
            & (np.abs(local_y) <= length / 2 + margin)
            & (np.abs(dz) <= height / 2 + margin)
        )
        keep &= ~inside

    return corners[keep]

def make_transform(rotation, translation):
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = Quaternion(rotation).rotation_matrix
    t[:3, 3] = np.asarray(translation, dtype=np.float64)
    return t


def ring_ordered_corners(points):
    """Run the OCaMo corner extractor separately on each ordered LiDAR ring."""
    # Remove rows with invalid coordinates, intensity, or ring values before
    # converting the ring column to integers or calling the corner extractor.
    ring_values = points[:, 4]
    valid_rows = (
        np.isfinite(points[:, :5]).all(axis=1)
        & (ring_values >= 0)
        & (ring_values < 32)
    )
    points = points[valid_rows]
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
        # The corner extractor can produce invalid values for degenerate scans.
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


def select_search_stable_support(
    corners,
    runtime_transform,
    intrinsic,
    width,
    height,
    roll_bound_rad,
    max_step_deg=SUPPORT_MAX_STEP_DEG,
):
    """Select runtime-visible corners that stay visible over the roll search.

    A point is retained only if it is in front of the camera and inside the
    usable image region at every sampled roll correction from -bound to
    +bound. The same returned points are then used to score all candidates.
    Sampling at 0.05 degrees makes the visibility envelope dense; it is a
    numerical approximation to continuous visibility over the search range.
    """
    corners = np.asarray(corners, dtype=np.float64)
    corners = corners[np.isfinite(corners).all(axis=1)]
    if len(corners) == 0:
        return corners

    max_step_rad = np.deg2rad(max_step_deg)
    intervals = max(1, int(np.ceil(2.0 * roll_bound_rad / max_step_rad)))
    roll_candidates = np.linspace(-roll_bound_rad, roll_bound_rad, intervals + 1)
    points_h = np.column_stack(
        [corners, np.ones(len(corners), dtype=np.float64)]
    )
    visible_for_all = np.ones(len(corners), dtype=bool)

    for roll_rad in roll_candidates:
        correction = np.eye(4, dtype=np.float64)
        correction[:3, :3] = Rotation.from_rotvec(
            np.array([roll_rad, 0.0, 0.0], dtype=np.float64)
        ).as_matrix()
        candidate_transform = runtime_transform @ correction
        camera_points = (candidate_transform @ points_h.T).T[:, :3]
        z = camera_points[:, 2]
        in_front = z > 0.1

        visible = np.zeros(len(corners), dtype=bool)
        if in_front.any():
            projected = (intrinsic @ camera_points[in_front].T).T
            pixels = projected[:, :2] / projected[:, 2:3]
            visible[in_front] = (
                np.isfinite(pixels).all(axis=1)
                & (pixels[:, 0] >= 0)
                & (pixels[:, 0] < width)
                & (pixels[:, 1] >= height / 3)
                & (pixels[:, 1] < height)
            )

        visible_for_all &= visible
        if not visible_for_all.any():
            break

    return corners[visible_for_all]

def select_common_search_support(
    corners,
    runtime_transforms,
    intrinsic,
    width,
    height,
    roll_bound_rad,
    max_step_deg=SUPPORT_MAX_STEP_DEG,
):
    """Select corners visible across every tested transform and roll candidate."""
    corners = np.asarray(corners, dtype=np.float64)
    corners = corners[np.isfinite(corners).all(axis=1)]

    if len(corners) == 0:
        return corners

    max_step_rad = np.deg2rad(max_step_deg)
    intervals = max(
        1, int(np.ceil(2.0 * roll_bound_rad / max_step_rad))
    )
    roll_candidates = np.linspace(
        -roll_bound_rad, roll_bound_rad, intervals + 1
    )
    points_h = np.column_stack(
        [corners, np.ones(len(corners), dtype=np.float64)]
    )
    visible_for_all = np.ones(len(corners), dtype=bool)

    for base_transform in runtime_transforms:
        for roll_rad in roll_candidates:
            correction = np.eye(4, dtype=np.float64)
            correction[:3, :3] = Rotation.from_rotvec(
                np.array([roll_rad, 0.0, 0.0], dtype=np.float64)
            ).as_matrix()

            candidate_transform = base_transform @ correction
            camera_points = (candidate_transform @ points_h.T).T[:, :3]
            z = camera_points[:, 2]
            in_front = z > 0.1

            visible = np.zeros(len(corners), dtype=bool)
            if in_front.any():
                projected = (
                    intrinsic @ camera_points[in_front].T
                ).T
                pixels = projected[:, :2] / projected[:, 2:3]
                visible[in_front] = (
                    np.isfinite(pixels).all(axis=1)
                    & (pixels[:, 0] >= 0)
                    & (pixels[:, 0] < width)
                    & (pixels[:, 1] >= height / 3)
                    & (pixels[:, 1] < height)
                )

            visible_for_all &= visible
            if not visible_for_all.any():
                return corners[:0]

    return corners[visible_for_all]


def add_local_tangents(points, neighbors=8):
    """Append a local 3D tangent estimated by PCA of nearby LiDAR corners."""
    xyz = np.asarray(points[:, :3], dtype=np.float64)
    count = len(xyz)
    tangents = np.zeros_like(xyz)
    if count >= 3:
        k = min(neighbors, count)
        _, indices = KDTree(xyz).query(xyz, k=k)
        if indices.ndim == 1:
            indices = indices[:, None]
        for i in range(count):
            local = xyz[indices[i, 1:]]
            if len(local) < 2:
                continue
            centered = local - local.mean(axis=0, keepdims=True)
            covariance = centered.T @ centered
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            tangent = eigenvectors[:, np.argmax(eigenvalues)]
            norm = np.linalg.norm(tangent)
            if np.isfinite(norm) and norm > 1e-8:
                tangents[i] = tangent / norm
    return np.column_stack([xyz, tangents])

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
    tangent_lidar = (
        correction @ pcl[:, 3:6].T
        if OBJECTIVE_MODE == "oriented_kernel"
        else None
    )
    points_camera = (
        transform[:3, :3] @ points_lidar
        + transform[:3, 3:4]
    )
    tangent_camera = (
        transform[:3, :3] @ tangent_lidar
        if tangent_lidar is not None
        else None
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
            distances, neighbor_indices = kdtree.query(xy, k=neighbor_count)
            if distances.ndim == 1:
                distances = distances[:, None]
                neighbor_indices = neighbor_indices[:, None]

            distance_similarity = np.exp(
                -(distances ** 2) / (2 * sigma ** 2)
            )

            if OBJECTIVE_MODE == "oriented_kernel":
                visible_tangents = tangent_camera[:, in_front][:, inside]
                next_camera_points = camera_points + 0.1 * visible_tangents
                next_projected = intrinsic @ next_camera_points
                next_depth = next_projected[2]
                valid_next = (
                    (next_depth > 0.1)
                    & np.isfinite(next_projected).all(axis=0)
                )
                next_pixels = np.zeros_like(next_projected[:2])
                next_pixels[:, valid_next] = (
                    next_projected[:2, valid_next]
                    / next_depth[valid_next]
                )
                tangent_uv = next_pixels - pixels[:, inside]
                lidar_tangent_angles = np.arctan2(
                    tangent_uv[1], tangent_uv[0]
                )

                edge_normals = current_frame["edge_normals"]
                edge_tangents = edge_normals[neighbor_indices] + np.pi / 2.0
                angle_difference = 0.5 * np.arctan2(
                    np.sin(2.0 * (
                        lidar_tangent_angles[:, None] - edge_tangents
                    )),
                    np.cos(2.0 * (
                        lidar_tangent_angles[:, None] - edge_tangents
                    )),
                )
                sigma_angle = np.deg2rad(ORIENTATION_SIGMA_DEG)
                orientation_similarity = np.exp(
                    -0.5 * (angle_difference / sigma_angle) ** 2
                )
                valid_tangent = (
                    valid_next
                    & (np.linalg.norm(tangent_uv, axis=0) > 1e-6)
                )
                orientation_similarity[~valid_tangent, :] = 0.0
                local_similarity = (
                    distance_similarity * orientation_similarity
                ).sum(axis=1)
            else:
                local_similarity = distance_similarity.sum(axis=1)
            similarity[visible_indices] = local_similarity

    if OBJECTIVE_MODE in {"kernel_sum", "oriented_kernel"}:
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
matched_seed_tag = (
    f"seed{MATCHED_SEED:02d}_"
    if SUPPORT_MODE == "runtime_matched"
    else ""
)
split_check_tag = "splitcheck_" if SPLIT_SUPPORT_CHECK else ""
output_path = Path(
    f"/workspace/bevfusion/results/calibration_baseline/"
    f"online_tracker_roll_{OBJECTIVE_MODE}_cam_{CAMERA_CHANNEL}_"
    f"inj_{injection_tag}_"
    f"support_{SUPPORT_MODE}_gtfilter_{int(USE_GT_MOVING_FILTER)}_"
    f"{matched_seed_tag}"
    f"{split_check_tag}"
    f"{scene['name']}.csv"
)
output_path.parent.mkdir(parents=True, exist_ok=True)

# Inject +0.5 degrees around the LiDAR-coordinate x axis.
injection = np.eye(4)
injection[:3, :3] = Rotation.from_rotvec(
    np.deg2rad([INJECTED_ROLL_DEG, 0.0, 0.0])
).as_matrix()

# Track only the roll-like x component. The expected correction is -0.5°.
tracker = sgd_module.SGDSchaul(
    params_trck=np.array([0]),
    theta_init=np.zeros(3),
    delta=np.array([0.001, 0.001, 0.001]),
    bnd=TRACKER_BOUNDS_RAD.copy(),
)

rows = []
# One reproducible RNG stream per run; each frame receives a new matched subset.
matched_rng = np.random.default_rng(MATCHED_SEED)
sample_token = scene["first_sample_token"]
frame_index = 0

while sample_token:
    sample = nusc.get("sample", sample_token)
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    camera_sd = nusc.get(
        "sample_data", sample["data"][CAMERA_CHANNEL]
    )

    lidar_path = Path(DATAROOT) / lidar_sd["filename"]
    camera_path = Path(DATAROOT) / camera_sd["filename"]

    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
    corners = ring_ordered_corners(points)
    if USE_GT_MOVING_FILTER:
        info = info_by_token[sample["token"]]
        corners = remove_moving_object_corners(corners, info)

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
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_normals = np.arctan2(grad_y[edge_y, edge_x], grad_x[edge_y, edge_x])
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
    # Freeze one support set per frame. "clean" is oracle support; "runtime"
    # uses visibility at the starting transform; "search_stable" uses only
    # the runtime transform and retains points visible across the full roll
    # correction bound. "runtime_matched" randomly downsamples runtime-visible
    # corners to the search-stable count as a count-matched control. No
    # candidate gets a different support set.
    if SUPPORT_MODE == "clean":
        support = select_initial_support(
            corners, clean_transform, intrinsic, width, height
        )

    elif SUPPORT_MODE == "runtime":
        support = select_initial_support(
            corners, bad_transform, intrinsic, width, height
        )

    elif SUPPORT_MODE == "runtime_matched":
        runtime_support = select_initial_support(
            corners, bad_transform, intrinsic, width, height
        )
        stable_support = select_search_stable_support(
            corners,
            bad_transform,
            intrinsic,
            width,
            height,
            roll_bound_rad=TRACKER_BOUNDS_RAD[0],
            max_step_deg=SUPPORT_MAX_STEP_DEG,
        )

        if len(stable_support) >= len(runtime_support):
            support = runtime_support
        else:
            selected = matched_rng.choice(
                len(runtime_support),
                size=len(stable_support),
                replace=False,
            )
            support = runtime_support[np.sort(selected)]

    elif SUPPORT_MODE == "search_stable":
        support = select_search_stable_support(
            corners,
            bad_transform,
            intrinsic,
            width,
            height,
            roll_bound_rad=TRACKER_BOUNDS_RAD[0],
            max_step_deg=SUPPORT_MAX_STEP_DEG,
        )

    else:  # shared_diagnostic
        # Controlled comparison set: clean and both known synthetic injections.
        # This mode is diagnostic and must not be described as deployable.
        shared_transforms = []

        for injection_deg in DIAGNOSTIC_INJECTIONS_DEG:
            known_injection = np.eye(4, dtype=np.float64)
            known_injection[:3, :3] = Rotation.from_rotvec(
                np.deg2rad([injection_deg, 0.0, 0.0])
            ).as_matrix()
            shared_transforms.append(clean_transform @ known_injection)

        support = select_common_search_support(
            corners,
            shared_transforms,
            intrinsic,
            width,
            height,
            roll_bound_rad=TRACKER_BOUNDS_RAD[0],
            max_step_deg=SUPPORT_MAX_STEP_DEG,
        )

    current_frame.update({
        "bad_transform": bad_transform,
        "intrinsic": intrinsic,
        "width": width,
        "height": height,
        "edge_normals": edge_normals,
    })

    objective_support = (
        add_local_tangents(support)
        if OBJECTIVE_MODE == "oriented_kernel"
        else support
    )
    
    # Diagnostic score grid (the support selector covers the wider optimizer
    # bound above, approximately +/-0.951 degrees).
    bound_deg = float(np.rad2deg(TRACKER_BOUNDS_RAD[0]))
    step_deg = 0.05
    steps = int(np.floor(bound_deg / step_deg))

    fixed_corrections_deg = sorted(set(
        [round(i * step_deg, 2) for i in range(-steps, steps + 1)]
        + [round(-bound_deg, 3), round(bound_deg, 3)]
    ))

    fixed_scores = {}
    for angle_deg in fixed_corrections_deg:
        params = np.deg2rad([angle_deg, 0.0, 0.0])
        score = -nuscenes_kernel_correlation(
            edge_tree, objective_support, params, 9, 10
        )
        fixed_scores[f"score_roll_{angle_deg:+.3f}"] = score

    # Optional held-out feature check. Each deterministic half proposes its
    # own grid peak; the other half must independently score that proposal
    # better than zero. The baseline tracker remains unchanged and is saved
    # separately below. In split-check mode, estimated_roll_deg is the
    # accepted split candidate, or zero when the check abstains.
    split_metrics = {
        "split_gate_accepted": "",
        "split_peak_a_deg": "",
        "split_peak_b_deg": "",
        "split_peak_agreement_deg": "",
        "split_gain_a_on_b_per_corner": "",
        "split_gain_b_on_a_per_corner": "",
        "split_gated_correction_deg": "",
    }
    split_correction_deg = 0.0
    if SPLIT_SUPPORT_CHECK:
        support_a = objective_support[::2]
        support_b = objective_support[1::2]
        split_accepted = False

        if len(support_a) > 0 and len(support_b) > 0:
            scores_a = np.asarray([
                -nuscenes_kernel_correlation(
                    edge_tree, support_a,
                    np.deg2rad([angle_deg, 0.0, 0.0]), 9, 10,
                )
                for angle_deg in fixed_corrections_deg
            ], dtype=np.float64)
            scores_b = np.asarray([
                -nuscenes_kernel_correlation(
                    edge_tree, support_b,
                    np.deg2rad([angle_deg, 0.0, 0.0]), 9, 10,
                )
                for angle_deg in fixed_corrections_deg
            ], dtype=np.float64)

            peak_a_idx = int(np.nanargmax(scores_a))
            peak_b_idx = int(np.nanargmax(scores_b))
            peak_a_deg = float(fixed_corrections_deg[peak_a_idx])
            peak_b_deg = float(fixed_corrections_deg[peak_b_idx])
            zero_idx = int(np.argmin(np.abs(fixed_corrections_deg)))

            # kernel_sum is additive; the other supported objectives already
            # return a per-point mean-like score.
            scale_a = len(support_a) if OBJECTIVE_MODE == "kernel_sum" else 1
            scale_b = len(support_b) if OBJECTIVE_MODE == "kernel_sum" else 1
            gain_a_on_b = float(
                (scores_b[peak_a_idx] - scores_b[zero_idx]) / scale_b
            )
            gain_b_on_a = float(
                (scores_a[peak_b_idx] - scores_a[zero_idx]) / scale_a
            )
            agreement_deg = abs(peak_a_deg - peak_b_deg)
            same_nonzero_sign = peak_a_deg * peak_b_deg > 0.0
            split_accepted = (
                same_nonzero_sign
                and agreement_deg <= SPLIT_PEAK_AGREEMENT_DEG
                and gain_a_on_b > 0.0
                and gain_b_on_a > 0.0
            )
            if split_accepted:
                split_correction_deg = (peak_a_deg + peak_b_deg) / 2.0

            split_metrics = {
                "split_gate_accepted": int(split_accepted),
                "split_peak_a_deg": peak_a_deg,
                "split_peak_b_deg": peak_b_deg,
                "split_peak_agreement_deg": agreement_deg,
                "split_gain_a_on_b_per_corner": gain_a_on_b,
                "split_gain_b_on_a_per_corner": gain_b_on_a,
                "split_gated_correction_deg": split_correction_deg,
            }

    if len(support) > 0:
        tracker.update(objective_support, edge_tree)

    raw_estimated_deg = float(np.rad2deg(tracker.theta_[0]))
    estimated_deg = (
        split_correction_deg if SPLIT_SUPPORT_CHECK else raw_estimated_deg
    )
    rows.append({
        "frame": frame_index,
        "sample_token": sample["token"],
        "scene_name": scene["name"],
        "camera": CAMERA_CHANNEL,
        "injected_roll_deg": INJECTED_ROLL_DEG,
        "expected_correction_deg": -INJECTED_ROLL_DEG,
        "estimated_roll_deg": estimated_deg,
        "tracker_estimate_raw_deg": raw_estimated_deg,
        "absolute_error_deg": abs(estimated_deg + INJECTED_ROLL_DEG),
        "support_corners": len(support),
        "matched_seed": MATCHED_SEED if SUPPORT_MODE == "runtime_matched" else "",
        "support_roll_bound_deg": float(np.rad2deg(TRACKER_BOUNDS_RAD[0])),
        "support_visibility_step_deg": SUPPORT_MAX_STEP_DEG,
        "camera_minus_lidar_ms":
            (camera_sd["timestamp"] - lidar_sd["timestamp"]) / 1000.0,
        "orientation_sigma_deg": (
            ORIENTATION_SIGMA_DEG
            if OBJECTIVE_MODE == "oriented_kernel"
            else ""
        ),
        **split_metrics,
        **fixed_scores,
    })

    if frame_index % 5 == 0:
        print(
            f"frame={frame_index:02d} "
            f"estimate={estimated_deg:+.3f} deg "
            f"expected={-INJECTED_ROLL_DEG:+.3f} deg"
            f"support={len(support)}"
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
    print("Final raw tracker estimate:",
          f"{rows[-1]['tracker_estimate_raw_deg']:+.3f} deg")
    print("Expected correction:", f"{-INJECTED_ROLL_DEG:+.3f} deg")
