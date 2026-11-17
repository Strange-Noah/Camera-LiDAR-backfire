from pathlib import Path
import sys

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

DATAROOT = "/workspace/nuscenes"
OCAMO_DIR = "/workspace/OCaMo/python/inference"
OUTPUT = Path(
    "/workspace/bevfusion/results/calibration_baseline/"
    "nuscenes_projection_check.jpg"
)

sys.path.insert(0, OCAMO_DIR)
from include.feature_extraction import get_corners


def make_transform(rotation, translation):
    """Build a 4x4 transform from a nuScenes quaternion and translation."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Quaternion(rotation).rotation_matrix
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


nusc = NuScenes(
    version="v1.0-trainval",
    dataroot=DATAROOT,
    verbose=False,
)

# Pick the first sample whose LiDAR and front-camera files are available.
chosen = None
for sample in nusc.sample:
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    camera_sd = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
    lidar_path = Path(DATAROOT) / lidar_sd["filename"]
    camera_path = Path(DATAROOT) / camera_sd["filename"]
    if lidar_path.is_file() and camera_path.is_file():
        chosen = (sample, lidar_sd, camera_sd, lidar_path, camera_path)
        break

if chosen is None:
    raise FileNotFoundError(
        "Could not find a sample with both LiDAR and CAM_FRONT files. "
        "Check that DATAROOT points to the complete nuScenes dataset."
    )

sample, lidar_sd, camera_sd, lidar_path, camera_path = chosen

# nuScenes point files store x, y, z, intensity, and ring index.
points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
image = cv2.imread(str(camera_path))
if image is None:
    raise RuntimeError(f"Could not read image: {camera_path}")

height, width = image.shape[:2]

# OCaMo's CARLA corner extractor expects a scan-ordered point array.
# This first check tests the current file order; it does not reorder points.
corners = get_corners(points)[:, :3]

lidar_calib = nusc.get(
    "calibrated_sensor", lidar_sd["calibrated_sensor_token"]
)
lidar_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
camera_calib = nusc.get(
    "calibrated_sensor", camera_sd["calibrated_sensor_token"]
)
camera_pose = nusc.get("ego_pose", camera_sd["ego_pose_token"])

# Transform LiDAR points through global coordinates into the camera frame.
T_lidar_to_global = (
    make_transform(lidar_pose["rotation"], lidar_pose["translation"])
    @ make_transform(lidar_calib["rotation"], lidar_calib["translation"])
)
T_camera_to_global = (
    make_transform(camera_pose["rotation"], camera_pose["translation"])
    @ make_transform(camera_calib["rotation"], camera_calib["translation"])
)
T_lidar_to_camera = np.linalg.inv(T_camera_to_global) @ T_lidar_to_global

points_h = np.column_stack(
    [corners, np.ones(len(corners), dtype=np.float64)]
)
points_camera = (T_lidar_to_camera @ points_h.T).T[:, :3]

intrinsic = np.asarray(camera_calib["camera_intrinsic"], dtype=np.float64)
in_front = points_camera[:, 2] > 0.1
projected = (intrinsic @ points_camera[in_front].T).T
pixels = projected[:, :2] / projected[:, 2:3]

inside = (
    (pixels[:, 0] >= 0)
    & (pixels[:, 0] < width)
    & (pixels[:, 1] >= 0)
    & (pixels[:, 1] < height)
)
visible_pixels = np.round(pixels[inside]).astype(np.int32)

# Compare projected corners with image edges in OCaMo's lower-image region.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 100)
edges[: height // 3] = 0
distance_to_edge = cv2.distanceTransform(
    255 - edges, cv2.DIST_L2, 3
)

if len(visible_pixels):
    distances = distance_to_edge[
        visible_pixels[:, 1], visible_pixels[:, 0]
    ]
    print(f"Sample: {sample['token']}")
    print(f"LiDAR points: {len(points)}")
    print(f"Ring values: {len(np.unique(points[:, 4]))}")
    print(f"OCaMo corners extracted: {len(corners)}")
    print(f"Corners in front of camera: {int(in_front.sum())}")
    print(f"Corners inside image: {len(visible_pixels)}")
    print(f"Median distance to image edge: {np.median(distances):.2f} px")
    print(f"Within 3 px of an edge: {(distances <= 3).mean():.1%}")
    print(f"Within 5 px of an edge: {(distances <= 5).mean():.1%}")
else:
    print(f"Sample: {sample['token']}")
    print(f"LiDAR points: {len(points)}")
    print(f"Ring values: {len(np.unique(points[:, 4]))}")
    print(f"OCaMo corners extracted: {len(corners)}")
    print(f"Corners in front of camera: {int(in_front.sum())}")
    print("No extracted corners projected inside the image.")

# Draw projected corners for a visual geometry check.
for x, y in visible_pixels:
    cv2.circle(image, (int(x), int(y)), 2, (0, 0, 255), -1)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(OUTPUT), image)
print(f"Projection overlay: {OUTPUT}")
print(
    "Camera/LiDAR timestamp difference: "
    f"{(camera_sd['timestamp'] - lidar_sd['timestamp']) / 1000:.1f} ms"
)
