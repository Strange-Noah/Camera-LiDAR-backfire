"""Exploratory scene-grouped regression from saved multi-camera score curves.

This script consumes objective-specific search_stable/no-GT-filter CSVs.
It does not modify the online tracker and its cross-validation results are
development evidence, not an independent final evaluation.
"""

from pathlib import Path
import os
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("results/calibration_baseline")
OBJECTIVE_MODE = os.environ.get("CURVE_OBJECTIVE_MODE", "kernel_sum")
CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)
CONDITIONS = {
    "plus0p00": 0.0,
    "plus0p50": 0.5,
    "minus0p50": -0.5,
}
BURN_IN_FRAME = 10
N_SPLITS = 5
RIDGE_ALPHA = 10.0


def input_path(camera, condition, scene_name):
    return ROOT / (
        f"online_tracker_roll_{OBJECTIVE_MODE}_cam_{camera}_"
        f"inj_{condition}_support_search_stable_gtfilter_0_"
        f"{scene_name}.csv"
    )


def discover_scenes():
    requested = os.environ.get("CURVE_SCENES")
    if requested:
        scenes = sorted({
            name.strip()
            for name in requested.split(",")
            if name.strip()
        })
        invalid = [name for name in scenes if not re.fullmatch(r"scene-\d+", name)]
        if invalid:
            raise ValueError(
                "CURVE_SCENES entries must look like scene-0011; invalid: "
                + ", ".join(invalid)
            )
        return scenes

    pattern = (
        f"online_tracker_roll_{OBJECTIVE_MODE}_cam_CAM_FRONT_"
        "inj_plus0p00_support_search_stable_gtfilter_0_scene-*.csv"
    )
    names = []
    for path in ROOT.glob(pattern):
        match = re.search(r"(scene-\d+)\.csv$", path.name)
        if match:
            names.append(match.group(1))
    return sorted(set(names))


def load_scene_condition(scene_name, condition, injection_deg, angle_columns):
    merged = None
    for camera in CAMERAS:
        path = input_path(camera, condition, scene_name)
        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)
        df = df[df["frame"] >= BURN_IN_FRAME].copy()
        score_cols = [c for c in df.columns if c.startswith("score_roll_")]
        if set(score_cols) != set(angle_columns):
            raise ValueError(f"Score grid differs in {path}")
        if "support_corners" not in df:
            raise ValueError(f"Missing support_corners in {path}")

        df = df.sort_values("frame")
        score_cols = list(angle_columns)
        score_matrix = df[score_cols].to_numpy(dtype=np.float64)
        zero_idx = score_cols.index("score_roll_+0.000")
        support_count = df["support_corners"].to_numpy(dtype=np.float64)
        normalized = (
            score_matrix - score_matrix[:, [zero_idx]]
        ) / np.maximum(support_count[:, None], 1.0)

        feature_names = [f"{camera}__{c}" for c in score_cols]
        camera_features = pd.DataFrame(
            normalized,
            columns=feature_names,
        )
        camera_features.insert(0, "sample_token", df["sample_token"].to_numpy())
        camera_features.insert(0, "frame", df["frame"].to_numpy())
        camera_features.insert(0, "scene_name", scene_name)
        camera_features.insert(0, "condition", condition)
        camera_features.insert(0, "target_correction_deg", -injection_deg)

        merged = (
            camera_features
            if merged is None
            else merged.merge(
                camera_features,
                on=[
                    "target_correction_deg",
                    "condition",
                    "scene_name",
                    "frame",
                    "sample_token",
                ],
                how="inner",
                validate="one_to_one",
            )
        )

    if merged is None or merged.empty:
        raise RuntimeError(f"No common camera frames for {scene_name}, {condition}")
    return merged


def main():
    if not ROOT.is_dir():
        raise FileNotFoundError(f"Results folder not found: {ROOT}")

    scenes = discover_scenes()
    if not scenes:
        raise FileNotFoundError(
            f"No CAM_FRONT clean search_stable files found under {ROOT}"
        )

    # Use a common score grid, and fail early if any required camera/condition
    # file is absent. Scene names, rather than numeric indices, key the runs.
    first_path = input_path("CAM_FRONT", "plus0p00", scenes[0])
    first_df = pd.read_csv(first_path, nrows=1)
    angle_columns = sorted(
        [c for c in first_df.columns if c.startswith("score_roll_")],
        key=lambda c: float(c.removeprefix("score_roll_")),
    )
    if "score_roll_+0.000" not in angle_columns:
        raise ValueError("Score grid is missing score_roll_+0.000")

    missing = []
    for scene_name in scenes:
        for condition in CONDITIONS:
            for camera in CAMERAS:
                path = input_path(camera, condition, scene_name)
                if not path.exists():
                    missing.append(str(path))
    if missing:
        preview = "\n".join(missing[:30])
        raise FileNotFoundError(
            f"Missing {len(missing)} required CSVs; first missing files:\n{preview}"
        )

    data = pd.concat(
        [
            load_scene_condition(scene, condition, injection, angle_columns)
            for scene in scenes
            for condition, injection in CONDITIONS.items()
        ],
        ignore_index=True,
    )

    metadata = [
        "target_correction_deg",
        "condition",
        "scene_name",
        "frame",
        "sample_token",
    ]
    feature_columns = [c for c in data.columns if c not in metadata]
    x = data[feature_columns].to_numpy(dtype=np.float64)
    y = data["target_correction_deg"].to_numpy(dtype=np.float64)
    groups = data["scene_name"].to_numpy()

    n_splits = min(N_SPLITS, len(np.unique(groups)))
    if n_splits < 2:
        raise RuntimeError("Need at least two complete scenes for grouped CV")

    # Every frame from one scene stays in the same fold. The scaler is fitted
    # inside each fold to avoid leaking held-out scene statistics.
    splitter = GroupKFold(n_splits=n_splits)
    predictions = np.full(len(data), np.nan, dtype=np.float64)
    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(x, y, groups=groups), start=1
    ):
        model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=RIDGE_ALPHA),
        )
        model.fit(x[train_idx], y[train_idx])
        predictions[test_idx] = model.predict(x[test_idx])
        print(
            f"fold {fold}/{n_splits}: train scenes="
            f"{len(np.unique(groups[train_idx]))}, test scenes="
            f"{len(np.unique(groups[test_idx]))}"
        )

    data["predicted_correction_deg"] = predictions
    data["absolute_error_deg"] = np.abs(
        predictions - data["target_correction_deg"].to_numpy()
    )
    data["within_0p10_deg"] = data["absolute_error_deg"] <= 0.10
    data["correct_direction"] = np.where(
        data["target_correction_deg"] > 0,
        predictions > 0,
        np.where(data["target_correction_deg"] < 0, predictions < 0, np.nan),
    )

    suffix = "" if OBJECTIVE_MODE == "kernel_sum" else f"_{OBJECTIVE_MODE}"
    detail_path = ROOT / f"score_curve_groupcv_predictions{suffix}.csv"
    data[metadata + [
        "predicted_correction_deg",
        "absolute_error_deg",
        "within_0p10_deg",
        "correct_direction",
    ]].to_csv(detail_path, index=False)

    per_scene = data.groupby(["scene_name", "condition"], as_index=False).agg(
        frames=("frame", "size"),
        mean_prediction_deg=("predicted_correction_deg", "mean"),
        mae_deg=("absolute_error_deg", "mean"),
        within_0p10=("within_0p10_deg", "mean"),
    )
    positive_target = data["target_correction_deg"] > 0
    negative_target = data["target_correction_deg"] < 0
    per_scene["correct_direction"] = np.nan
    for (scene_name, condition), idx in data.groupby(
        ["scene_name", "condition"]
    ).groups.items():
        target = float(data.loc[idx[0], "target_correction_deg"])
        if target > 0:
            per_scene.loc[
                (per_scene["scene_name"] == scene_name)
                & (per_scene["condition"] == condition),
                "correct_direction",
            ] = data.loc[idx, "correct_direction"].mean()
        elif target < 0:
            per_scene.loc[
                (per_scene["scene_name"] == scene_name)
                & (per_scene["condition"] == condition),
                "correct_direction",
            ] = data.loc[idx, "correct_direction"].mean()

    summary_path = ROOT / f"score_curve_groupcv_scene_metrics{suffix}.csv"
    per_scene.to_csv(summary_path, index=False)

    # Exploratory abstention curves at predeclared absolute-correction
    # thresholds. These are not a frozen gate: selecting a threshold here
    # requires a later evaluation on entirely new scenes.
    gate_rows = []
    for cutoff_deg in (0.10, 0.20, 0.336):
        for (scene_name, condition), part in data.groupby(
            ["scene_name", "condition"]
        ):
            target = float(part["target_correction_deg"].iloc[0])
            pred = part["predicted_correction_deg"].to_numpy()
            accepted = np.abs(pred) >= cutoff_deg
            row = {
                "scene_name": scene_name,
                "condition": condition,
                "cutoff_deg": cutoff_deg,
                "frames": len(part),
                "accepted_frames": int(accepted.sum()),
                "coverage_or_clean_false_rate": float(accepted.mean()),
            }
            if target != 0.0 and accepted.any():
                chosen = pred[accepted]
                row["correct_direction"] = float(
                    np.mean(chosen < 0 if target < 0 else chosen > 0)
                )
                row["within_0p10"] = float(
                    np.mean(np.abs(chosen - target) <= 0.10)
                )
                row["accepted_mae_deg"] = float(
                    np.mean(np.abs(chosen - target))
                )
            else:
                row.update({
                    "correct_direction": np.nan,
                    "within_0p10": np.nan,
                    "accepted_mae_deg": np.nan,
                })
            gate_rows.append(row)

    gate_sweep = pd.DataFrame(gate_rows)
    gate_path = ROOT / f"score_curve_groupcv_gate_sweep{suffix}.csv"
    gate_sweep.to_csv(gate_path, index=False)

    print(f"\nComplete scenes: {len(scenes)}")
    print(f"Samples after burn-in: {len(data)}")
    print(f"Score-curve features: {len(feature_columns)}")
    print(f"Ridge alpha: {RIDGE_ALPHA}")
    print(f"Saved OOF predictions: {detail_path}")
    print(f"Saved per-scene metrics: {summary_path}")
    print(f"Saved exploratory abstention sweep: {gate_path}")
    print("\nEqual-scene means (exploratory grouped CV):")
    print(
        per_scene.groupby("condition")[
            ["mean_prediction_deg", "mae_deg", "within_0p10", "correct_direction"]
        ].mean(numeric_only=True).to_string(float_format=lambda v: f"{v:.3f}")
    )
    print("\nExploratory abstention sweep (equal-scene means):")
    print(
        gate_sweep.groupby(["condition", "cutoff_deg"])[
            [
                "coverage_or_clean_false_rate",
                "correct_direction",
                "within_0p10",
                "accepted_mae_deg",
            ]
        ].mean(numeric_only=True).to_string(
            float_format=lambda v: f"{v:.3f}"
        )
    )


if __name__ == "__main__":
    main()
