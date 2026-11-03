#!/usr/bin/env python3
"""Scene-cluster bootstrap for selected fusion-vs-LiDAR recall groups."""
import argparse
import os
import numpy as np
import pandas as pd

FUSION = ["minus300", "minus200", "minus100", "zeroms", "plus100", "plus200", "plus300"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matches", required=True, help="per_gt_matches.csv")
    p.add_argument("--out", required=True, help="Output CSV path")
    p.add_argument("--classes", nargs="+", default=["motorcycle", "truck", "bicycle"])
    p.add_argument("--speed-bins", nargs="+", default=["5mps+", "1-5mps"])
    p.add_argument("--distance-bins", nargs="+", default=["0-20m", "20-40m", "40-60m", "60m+"])
    p.add_argument("--score-thresholds", nargs="+", type=float, default=[0.1, 0.3, 0.5])
    p.add_argument("--center-threshold", type=float, default=2.0)
    p.add_argument("--reference-condition", choices=FUSION + ["lidar_only"],
                   default="lidar_only",
                   help="Compare each other condition against this reference.")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--chunksize", type=int, default=250000)
    a = p.parse_args()

    wanted_cols = ["condition", "scene_token", "score_threshold",
                   "center_threshold_m", "class_name", "speed_bin",
                   "distance_bin", "matched"]
    agg = None
    reader = pd.read_csv(a.matches, usecols=wanted_cols, chunksize=a.chunksize)
    for chunk in reader:
        chunk = chunk[
            chunk["class_name"].isin(a.classes)
            & chunk["speed_bin"].isin(a.speed_bins)
            & chunk["distance_bin"].isin(a.distance_bins)
            & np.isclose(chunk["center_threshold_m"], a.center_threshold)
            & chunk["score_threshold"].round(6).isin([round(x, 6) for x in a.score_thresholds])
        ]
        if chunk.empty:
            continue
        part = (chunk.groupby(wanted_cols[:-1], observed=True)["matched"]
                .agg(gt="size", tp="sum").reset_index())
        agg = part if agg is None else pd.concat([agg, part], ignore_index=True)

    if agg is None or agg.empty:
        raise SystemExit("No rows matched the requested filters. Check bin labels and thresholds.")
    agg = (agg.groupby(wanted_cols[:-1], as_index=False, observed=True)
           [["gt", "tp"]].sum())

    rng = np.random.default_rng(a.seed)
    output = []
    group_cols = ["score_threshold", "center_threshold_m", "class_name", "speed_bin", "distance_bin"]
    for group_key, group in agg.groupby(group_cols, observed=True):
        score, center, cls, speed, distance = group_key
        pivot_gt = group.pivot(index="scene_token", columns="condition", values="gt").fillna(0)
        pivot_tp = group.pivot(index="scene_token", columns="condition", values="tp").fillna(0)
        reference = a.reference_condition
        conditions = [c for c in FUSION + ["lidar_only"]
                      if c != reference and c in pivot_gt.columns]
        if reference not in pivot_gt.columns:
            continue
        scenes = pivot_gt.index.to_numpy()
        if len(scenes) < 2:
            continue
        gt_arr = pivot_gt.reindex(columns=[reference] + conditions, fill_value=0).to_numpy(float)
        tp_arr = pivot_tp.reindex(columns=[reference] + conditions, fill_value=0).to_numpy(float)
        draws = rng.integers(0, len(scenes), size=(a.bootstrap, len(scenes)))
        base_gt = gt_arr[:, 0][draws].sum(axis=1)
        base_tp = tp_arr[:, 0][draws].sum(axis=1)
        base_recall = np.divide(base_tp, base_gt, out=np.full(a.bootstrap, np.nan), where=base_gt > 0)
        lidar_recall = float(tp_arr[:, 0].sum() / gt_arr[:, 0].sum()) if gt_arr[:, 0].sum() else np.nan
        for ci, condition in enumerate(conditions, start=1):
            cond_gt = gt_arr[:, ci][draws].sum(axis=1)
            cond_tp = tp_arr[:, ci][draws].sum(axis=1)
            cond_recall = np.divide(cond_tp, cond_gt, out=np.full(a.bootstrap, np.nan), where=cond_gt > 0)
            delta = cond_recall - base_recall
            delta = delta[np.isfinite(delta)]
            full_recall = float(tp_arr[:, ci].sum() / gt_arr[:, ci].sum()) if gt_arr[:, ci].sum() else np.nan
            scene_mask = gt_arr[:, 0] > 0
            output.append({
                "class_name": cls, "speed_bin": speed, "distance_bin": distance,
                "score_threshold": score, "center_threshold_m": center,
                "condition": condition, "reference_condition": reference,
                "scenes": int(scene_mask.sum()),
                "gt_objects": int(gt_arr[:, 0].sum()),
                "fusion_recall": full_recall, "lidar_recall": lidar_recall,
                "recall_difference": full_recall - lidar_recall,
                "bootstrap_ci95_low": float(np.quantile(delta, 0.025)) if len(delta) else np.nan,
                "bootstrap_ci95_high": float(np.quantile(delta, 0.975)) if len(delta) else np.nan,
                "bootstrap_replicates": len(delta),
            })

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    result = pd.DataFrame(output)
    result.to_csv(a.out, index=False)
    print(f"Reference condition: {a.reference_condition}")
    print(f"Wrote {len(result)} scene-bootstrap comparisons to {a.out}")
    if not result.empty:
        print(result.sort_values("recall_difference").head(30).to_string(index=False))


if __name__ == "__main__":
    main()
