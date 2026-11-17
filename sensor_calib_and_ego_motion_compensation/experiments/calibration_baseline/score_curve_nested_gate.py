"""Nested scene-grouped development test for score-curve ridge calibration.

All hyperparameter and abstention choices are made using inner scene-grouped
folds. Outer folds estimate performance on scenes excluded from those choices.
Because these scenes have already been examined in this project, results remain
exploratory and require a later untouched-scene evaluation.
"""

from pathlib import Path
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from score_curve_groupcv import (
    BURN_IN_FRAME,
    CAMERAS,
    CONDITIONS,
    OBJECTIVE_MODE,
    ROOT,
    discover_scenes,
    input_path,
    load_scene_condition,
)


OUTER_SPLITS = 5
INNER_SPLITS = 4
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
OUTPUT_GAINS = (0.8, 1.0, 1.2, 1.4, 1.6)
ABSTENTION_CUTOFFS = tuple(np.round(np.arange(0.10, 0.901, 0.01), 2))

# Recommended initial screening criteria. Change these only as a documented
# development decision; the outer folds are still not a final untouched test.
MAX_CLEAN_FALSE_SCENE_RATE = 0.05
MIN_INJECTION_COVERAGE = 0.10
MIN_DIRECTION_ACCURACY = 0.95
MAX_ACCEPTED_MAE_DEG = 0.10


def make_model(alpha):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def grouped_oof_predictions(x, y, groups, alpha, n_splits):
    unique_groups = np.unique(groups)
    n_splits = min(n_splits, len(unique_groups))
    if n_splits < 2:
        raise RuntimeError("Need at least two scenes for grouped CV")

    predictions = np.full(len(y), np.nan, dtype=np.float64)
    splitter = GroupKFold(n_splits=n_splits)
    for train_idx, valid_idx in splitter.split(x, y, groups=groups):
        model = make_model(alpha)
        model.fit(x[train_idx], y[train_idx])
        predictions[valid_idx] = model.predict(x[valid_idx])
    return predictions


def operating_metrics(frame, prediction, gain, cutoff):
    scored = frame[["scene_name", "condition", "target_correction_deg"]].copy()
    scored["prediction"] = prediction * gain
    scored["accepted"] = scored["prediction"].abs() >= cutoff

    per_scene = []
    for (scene, condition), part in scored.groupby(["scene_name", "condition"]):
        target = float(part["target_correction_deg"].iloc[0])
        accepted = part[part["accepted"]]
        row = {
            "scene_name": scene,
            "condition": condition,
            "coverage": float(part["accepted"].mean()),
            "any_accept": bool(part["accepted"].any()),
        }
        if target == 0.0:
            row["clean_false_scene"] = row["any_accept"]
        elif len(accepted):
            values = accepted["prediction"].to_numpy()
            row["direction"] = float(
                np.mean(values < 0 if target < 0 else values > 0)
            )
            row["accepted_mae_deg"] = float(np.mean(np.abs(values - target)))
        else:
            row["direction"] = np.nan
            row["accepted_mae_deg"] = np.nan
        per_scene.append(row)

    per_scene = pd.DataFrame(per_scene)
    clean = per_scene[per_scene["condition"] == "plus0p00"]
    injections = {}
    for condition in ("plus0p50", "minus0p50"):
        part = per_scene[per_scene["condition"] == condition]
        injections[condition] = {
            "coverage": float(part["coverage"].mean()),
            "direction": float(part["direction"].mean()),
            "mae": float(part["accepted_mae_deg"].mean()),
        }
    return {
        "clean_false_scene_rate": float(clean["clean_false_scene"].mean()),
        "injections": injections,
        "per_scene": per_scene,
    }


def choose_setting(frame, inner_predictions):
    candidates = []
    for alpha, prediction in inner_predictions.items():
        for gain in OUTPUT_GAINS:
            for cutoff in ABSTENTION_CUTOFFS:
                metrics = operating_metrics(frame, prediction, gain, cutoff)
                clean_rate = metrics["clean_false_scene_rate"]
                plus = metrics["injections"]["plus0p50"]
                minus = metrics["injections"]["minus0p50"]

                feasible = (
                    clean_rate <= MAX_CLEAN_FALSE_SCENE_RATE
                    and plus["coverage"] >= MIN_INJECTION_COVERAGE
                    and minus["coverage"] >= MIN_INJECTION_COVERAGE
                    and plus["direction"] >= MIN_DIRECTION_ACCURACY
                    and minus["direction"] >= MIN_DIRECTION_ACCURACY
                    and plus["mae"] <= MAX_ACCEPTED_MAE_DEG
                    and minus["mae"] <= MAX_ACCEPTED_MAE_DEG
                )
                if not feasible:
                    continue

                # Maximize the weaker injection coverage after satisfying all
                # screening criteria; use MAE and clean rate as tie breakers.
                candidates.append((
                    min(plus["coverage"], minus["coverage"]),
                    -(plus["mae"] + minus["mae"]) / 2.0,
                    -clean_rate,
                    alpha,
                    gain,
                    cutoff,
                ))

    if not candidates:
        return None
    best = max(candidates)
    return {
        "alpha": best[3],
        "gain": best[4],
        "cutoff": best[5],
        "inner_min_coverage": best[0],
        "inner_mean_mae": -best[1],
        "inner_clean_false_scene_rate": -best[2],
    }


def load_dataset():
    scenes = discover_scenes()
    if not scenes:
        raise FileNotFoundError(f"No clean CAM_FRONT runs found under {ROOT}")

    first = pd.read_csv(input_path("CAM_FRONT", "plus0p00", scenes[0]), nrows=1)
    angle_columns = sorted(
        [c for c in first.columns if c.startswith("score_roll_")],
        key=lambda c: float(c.removeprefix("score_roll_")),
    )
    if "score_roll_+0.000" not in angle_columns:
        raise ValueError("Score grid is missing score_roll_+0.000")

    missing = [
        str(input_path(camera, condition, scene))
        for scene in scenes
        for condition in CONDITIONS
        for camera in CAMERAS
        if not input_path(camera, condition, scene).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} required CSVs; first files:\n"
            + "\n".join(missing[:30])
        )

    data = pd.concat([
        load_scene_condition(scene, condition, injection, angle_columns)
        for scene in scenes
        for condition, injection in CONDITIONS.items()
    ], ignore_index=True)
    meta = {
        "target_correction_deg", "condition", "scene_name", "frame", "sample_token"
    }
    feature_columns = [column for column in data.columns if column not in meta]
    return data, feature_columns


def main():
    data, feature_columns = load_dataset()
    x = data[feature_columns].to_numpy(dtype=np.float64)
    y = data["target_correction_deg"].to_numpy(dtype=np.float64)
    groups = data["scene_name"].to_numpy()
    outer = GroupKFold(n_splits=min(OUTER_SPLITS, len(np.unique(groups))))

    result = data[[
        "scene_name", "condition", "frame", "sample_token",
        "target_correction_deg",
    ]].copy()
    result["raw_prediction_deg"] = np.nan
    result["selected_alpha"] = np.nan
    result["selected_gain"] = np.nan
    result["selected_cutoff_deg"] = np.nan
    result["accepted"] = False
    result["applied_correction_deg"] = 0.0
    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(
        outer.split(x, y, groups=groups), start=1
    ):
        train_groups = groups[train_idx]
        inner_predictions = {}
        for alpha in RIDGE_ALPHAS:
            inner_predictions[alpha] = grouped_oof_predictions(
                x[train_idx],
                y[train_idx],
                train_groups,
                alpha,
                INNER_SPLITS,
            )

        setting = choose_setting(data.iloc[train_idx], inner_predictions)
        if setting is None:
            print(f"outer fold {fold}: no inner setting met all criteria; abstaining")
            model = make_model(RIDGE_ALPHAS[0])
            model.fit(x[train_idx], y[train_idx])
            raw = model.predict(x[test_idx])
            result.loc[test_idx, "raw_prediction_deg"] = raw
            fold_rows.append({
                "fold": fold,
                "test_scenes": ",".join(sorted(set(groups[test_idx]))),
                "feasible_setting": False,
            })
            continue

        model = make_model(setting["alpha"])
        model.fit(x[train_idx], y[train_idx])
        raw = model.predict(x[test_idx])
        applied = raw * setting["gain"]
        accepted = np.abs(applied) >= setting["cutoff"]

        result.loc[test_idx, "raw_prediction_deg"] = raw
        result.loc[test_idx, "selected_alpha"] = setting["alpha"]
        result.loc[test_idx, "selected_gain"] = setting["gain"]
        result.loc[test_idx, "selected_cutoff_deg"] = setting["cutoff"]
        result.loc[test_idx, "accepted"] = accepted
        result.loc[test_idx, "applied_correction_deg"] = np.where(
            accepted, applied, 0.0
        )
        fold_rows.append({
            "fold": fold,
            "test_scenes": ",".join(sorted(set(groups[test_idx]))),
            "feasible_setting": True,
            **setting,
        })
        print(
            f"outer fold {fold}: alpha={setting['alpha']}, "
            f"gain={setting['gain']:.2f}, cutoff={setting['cutoff']:.2f}° "
            f"(selected inside training scenes only)"
        )

    suffix = "" if OBJECTIVE_MODE == "kernel_sum" else f"_{OBJECTIVE_MODE}"
    run_tag = os.environ.get("CURVE_RUN_TAG", "").strip()
    if run_tag:
        if not all(char.isalnum() or char in "-_" for char in run_tag):
            raise ValueError("CURVE_RUN_TAG may contain only letters, digits, '-' and '_'.")
        suffix += f"_{run_tag}"
    out = ROOT / f"score_curve_nested_gate{suffix}_oof.csv"
    result.to_csv(out, index=False)
    fold_out = ROOT / f"score_curve_nested_gate{suffix}_settings.csv"
    pd.DataFrame(fold_rows).to_csv(fold_out, index=False)

    scene_rows = []
    for (scene_name, condition), part in result.groupby(
        ["scene_name", "condition"]
    ):
        target = float(part["target_correction_deg"].iloc[0])
        accepted = part[part["accepted"]]
        row = {
            "scene_name": scene_name,
            "condition": condition,
            "frames": len(part),
            "accepted_frames": len(accepted),
            "coverage": float(part["accepted"].mean()),
        }
        if target == 0.0:
            row["clean_false_scene"] = bool(part["accepted"].any())
        elif len(accepted):
            values = accepted["applied_correction_deg"].to_numpy()
            row["correct_direction"] = float(
                np.mean(values < 0 if target < 0 else values > 0)
            )
            row["within_0p10"] = float(
                np.mean(np.abs(values - target) <= 0.10)
            )
            row["accepted_mae_deg"] = float(
                np.mean(np.abs(values - target))
            )
        else:
            row.update({
                "correct_direction": np.nan,
                "within_0p10": np.nan,
                "accepted_mae_deg": np.nan,
            })
        scene_rows.append(row)
    scene_metrics = pd.DataFrame(scene_rows)
    scene_out = ROOT / f"score_curve_nested_gate{suffix}_scene_metrics.csv"
    scene_metrics.to_csv(scene_out, index=False)

    print(f"\nComplete scenes: {len(np.unique(groups))}")
    print(f"Rows: {len(result)}; features: {len(feature_columns)}")
    print(f"Saved outer-fold predictions: {out}")
    print(f"Saved selected fold settings: {fold_out}")
    print(f"Saved outer-fold scene metrics: {scene_out}")
    print(
        "Feasible outer folds: "
        f"{sum(row['feasible_setting'] for row in fold_rows)}/{len(fold_rows)}"
    )
    print(
        "Screening criteria: clean false-scene rate <= "
        f"{MAX_CLEAN_FALSE_SCENE_RATE:.0%}; both injection coverages >= "
        f"{MIN_INJECTION_COVERAGE:.0%}; direction >= "
        f"{MIN_DIRECTION_ACCURACY:.0%}; accepted MAE <= "
        f"{MAX_ACCEPTED_MAE_DEG:.2f} degrees"
    )
    print("\nOuter-fold results (equal-scene means):")
    print(
        scene_metrics.groupby("condition")[
            [
                "coverage",
                "clean_false_scene",
                "correct_direction",
                "within_0p10",
                "accepted_mae_deg",
            ]
        ].mean(numeric_only=True).to_string(
            float_format=lambda value: f"{value:.3f}"
        )
    )


if __name__ == "__main__":
    main()
