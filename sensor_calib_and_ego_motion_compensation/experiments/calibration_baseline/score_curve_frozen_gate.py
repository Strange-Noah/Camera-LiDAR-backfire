"""Fit a development-only score-curve gate and evaluate it unchanged on holdout scenes.

Usage (from /workspace/bevfusion):
  FROZEN_GATE_MODE=fit CURVE_OBJECTIVE_MODE=kernel_sum \
    python3 experiments/calibration_baseline/score_curve_frozen_gate.py

  FROZEN_GATE_MODE=evaluate CURVE_OBJECTIVE_MODE=kernel_sum \
  CURVE_SCENES=scene-0080,scene-0081 FROZEN_GATE_TAG=holdout_0080_0081 \
    python3 experiments/calibration_baseline/score_curve_frozen_gate.py

CURVE_SCENES selects the input scenes in either mode. If omitted during fit,
all discovered scenes are used. It is required during evaluation. The holdout
scenes must not have been used for development or gate selection.
"""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from score_curve_groupcv import (
    OBJECTIVE_MODE,
    ROOT,
    discover_scenes,
)
from score_curve_nested_gate import (
    RIDGE_ALPHAS,
    choose_setting,
    grouped_oof_predictions,
    load_dataset,
    make_model,
)


DEFAULT_MODEL_PATH = ROOT / f"frozen_{OBJECTIVE_MODE}_gate.joblib"


def selected_scenes():
    requested = os.environ.get("CURVE_SCENES", "").strip()
    if requested:
        return sorted({name.strip() for name in requested.split(",") if name.strip()})
    return discover_scenes()


def fit_gate(model_path):
    scenes = selected_scenes()
    if len(scenes) < 5:
        raise ValueError(
            f"Need at least five development scenes for grouped selection; got {len(scenes)}"
        )

    data, feature_columns = load_dataset()
    groups = data["scene_name"].to_numpy()
    x = data[feature_columns].to_numpy(dtype=np.float64)
    y = data["target_correction_deg"].to_numpy(dtype=np.float64)

    # Each development scene gets predictions from a model that did not train
    # on that scene. Use these out-of-fold predictions to select one fixed gate.
    oof_predictions = {}
    for alpha in RIDGE_ALPHAS:
        oof_predictions[alpha] = grouped_oof_predictions(
            x, y, groups, alpha, n_splits=5
        )

    setting = choose_setting(data, oof_predictions)
    if setting is None:
        raise RuntimeError(
            "No alpha/gain/cutoff met all development screening criteria. "
            "Do not evaluate a deployable gate; revise using development data."
        )

    model = make_model(setting["alpha"])
    model.fit(x, y)
    artifact = {
        "objective_mode": OBJECTIVE_MODE,
        "development_scenes": scenes,
        "feature_columns": feature_columns,
        "alpha": setting["alpha"],
        "gain": setting["gain"],
        "cutoff_deg": setting["cutoff"],
        "inner_min_coverage": setting["inner_min_coverage"],
        "inner_mean_mae_deg": setting["inner_mean_mae"],
        "inner_clean_false_scene_rate": setting["inner_clean_false_scene_rate"],
        "model": model,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)

    summary_path = model_path.with_suffix(".json")
    summary = {key: value for key, value in artifact.items() if key != "model"}
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Development scenes: {len(scenes)} ({', '.join(scenes)})")
    print(
        "Frozen gate: "
        f"alpha={setting['alpha']}, gain={setting['gain']:.2f}, "
        f"cutoff={setting['cutoff']:.2f} degrees"
    )
    print(
        "Development OOF screening: "
        f"clean false-scene rate={setting['inner_clean_false_scene_rate']:.1%}, "
        f"minimum injection coverage={setting['inner_min_coverage']:.1%}, "
        f"mean accepted MAE={setting['inner_mean_mae']:.3f} degrees"
    )
    print(f"Saved model: {model_path}")
    print(f"Saved settings: {summary_path}")


def evaluate_gate(model_path):
    requested = os.environ.get("CURVE_SCENES", "").strip()
    if not requested:
        raise ValueError(
            "Set CURVE_SCENES to an explicit comma-separated holdout scene list."
        )
    scenes = selected_scenes()
    artifact = joblib.load(model_path)
    if artifact["objective_mode"] != OBJECTIVE_MODE:
        raise ValueError(
            f"Model objective is {artifact['objective_mode']}, "
            f"but CURVE_OBJECTIVE_MODE is {OBJECTIVE_MODE}."
        )
    overlap = sorted(set(scenes) & set(artifact["development_scenes"]))
    if overlap:
        raise ValueError(
            "Holdout scenes overlap development scenes: " + ", ".join(overlap)
        )

    data, feature_columns = load_dataset()
    if feature_columns != artifact["feature_columns"]:
        raise ValueError("Holdout feature columns/order do not match the saved model.")

    x = data[feature_columns].to_numpy(dtype=np.float64)
    prediction = artifact["model"].predict(x) * float(artifact["gain"])
    cutoff = float(artifact["cutoff_deg"])
    accepted = np.abs(prediction) >= cutoff
    result = data[[
        "scene_name", "condition", "frame", "sample_token",
        "target_correction_deg",
    ]].copy()
    result["predicted_correction_deg"] = prediction
    result["accepted"] = accepted
    result["applied_correction_deg"] = np.where(accepted, prediction, 0.0)

    scene_rows = []
    for (scene, condition), part in result.groupby(["scene_name", "condition"]):
        target = float(part["target_correction_deg"].iloc[0])
        selected = part[part["accepted"]]
        row = {
            "scene_name": scene,
            "condition": condition,
            "frames": len(part),
            "accepted_frames": len(selected),
            "coverage": float(part["accepted"].mean()),
        }
        if target == 0.0:
            row["clean_false_scene"] = bool(part["accepted"].any())
        elif len(selected):
            values = selected["applied_correction_deg"].to_numpy()
            row["correct_direction"] = float(
                np.mean(values < 0 if target < 0 else values > 0)
            )
            row["within_0p10"] = float(np.mean(np.abs(values - target) <= 0.10))
            row["accepted_mae_deg"] = float(np.mean(np.abs(values - target)))
        else:
            row.update({
                "correct_direction": np.nan,
                "within_0p10": np.nan,
                "accepted_mae_deg": np.nan,
            })
        scene_rows.append(row)

    tag = os.environ.get("FROZEN_GATE_TAG", "holdout").strip()
    if not tag or not all(char.isalnum() or char in "-_" for char in tag):
        raise ValueError("FROZEN_GATE_TAG may contain only letters, digits, '-' and '_'.")
    predictions_path = ROOT / f"frozen_{OBJECTIVE_MODE}_gate_{tag}_predictions.csv"
    metrics_path = ROOT / f"frozen_{OBJECTIVE_MODE}_gate_{tag}_scene_metrics.csv"
    metrics = pd.DataFrame(scene_rows)
    result.to_csv(predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    clean = metrics[metrics["condition"] == "plus0p00"]
    false_scenes = int(clean["clean_false_scene"].sum())
    print(f"Holdout scenes: {len(scenes)} ({', '.join(scenes)})")
    print(
        f"Frozen gate: alpha={artifact['alpha']}, gain={artifact['gain']:.2f}, "
        f"cutoff={cutoff:.2f} degrees"
    )
    print(f"Clean false-trigger scenes: {false_scenes}/{len(clean)}")
    for condition in ("plus0p50", "minus0p50"):
        part = metrics[metrics["condition"] == condition]
        print(f"{condition}, equal-scene means:")
        print(part[[
            "coverage", "correct_direction", "within_0p10", "accepted_mae_deg"
        ]].mean(numeric_only=True).round(3).to_string())
    print(f"Saved predictions: {predictions_path}")
    print(f"Saved scene metrics: {metrics_path}")


def main():
    mode = os.environ.get("FROZEN_GATE_MODE", "").strip().lower()
    model_path = Path(os.environ.get("FROZEN_GATE_PATH", str(DEFAULT_MODEL_PATH)))
    if mode == "fit":
        fit_gate(model_path)
    elif mode == "evaluate":
        evaluate_gate(model_path)
    else:
        raise ValueError("Set FROZEN_GATE_MODE to 'fit' or 'evaluate'.")


if __name__ == "__main__":
    main()
