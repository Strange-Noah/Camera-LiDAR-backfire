#!/usr/bin/env python3
"""Compare BEVFusion and LiDAR-only detections by GT speed and distance.

This analysis uses fixed-score operating points and greedy, score-ordered,
same-class XY-center matching. It is a stratified diagnostic, not nuScenes AP.
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
from mmengine.fileio import load


CLASSES = (
    "car", "truck", "bus", "trailer", "construction_vehicle",
    "pedestrian", "motorcycle", "bicycle", "traffic_cone", "barrier",
)
NAME_MAPPING = {
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.construction": "construction_vehicle",
}
CONDITIONS = (
    "minus300", "minus200", "minus100", "zeroms",
    "plus100", "plus200", "plus300", "lidar_only",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--info", required=True)
    parser.add_argument("--common-tokens", required=True)
    parser.add_argument("--manifest", required=True,
                        help="Use the zero-offset manifest to map sample tokens to scenes.")
    parser.add_argument("--results-dir", required=True,
                        help="Directory containing <condition>/predictions.pkl")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--score-thresholds", nargs="+", type=float,
                        default=[0.1, 0.3, 0.5])
    parser.add_argument("--center-thresholds", nargs="+", type=float,
                        default=[0.5, 1.0, 2.0, 4.0])
    return parser.parse_args()


def read_scene_map(path, common):
    scene_by_token = {}
    with open(path, newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            token = row["sample_token"]
            if token in common:
                scene_by_token[token] = row["scene_token"]
    missing = common.difference(scene_by_token)
    if missing:
        raise ValueError(f"Manifest has no scene mapping for {len(missing)} common samples")
    return scene_by_token


def get_gt(sample):
    names = sample["gt_names"]
    boxes = np.asarray(sample["gt_boxes"])
    velocities = np.asarray(sample["gt_velocity"])
    lidar_pts = np.asarray(sample["num_lidar_pts"])
    if not (len(names) == len(boxes) == len(velocities) == len(lidar_pts)):
        raise ValueError(f"Misaligned GT arrays for sample {sample['token']}")

    objects = []
    for idx, (raw_name, box, velocity, num_pts) in enumerate(
            zip(names, boxes, velocities, lidar_pts)):
        # Match this repository's NuScenesDataset.get_ann_info filtering:
        # use_valid_flag=False means keep boxes with at least one LiDAR point.
        if num_pts <= 0:
            continue
        name = NAME_MAPPING.get(str(raw_name), str(raw_name))
        if name not in CLASSES:
            continue
        xy = np.asarray(box[:2], dtype=np.float64)
        speed = float(np.linalg.norm(velocity)) if np.isfinite(velocity).all() else np.nan
        objects.append({
            "gt_index": idx,
            "class_name": name,
            "x": float(xy[0]),
            "y": float(xy[1]),
            "distance_m": float(np.linalg.norm(xy)),
            "speed_mps": speed,
        })
    return objects


def dist_bin(distance):
    if distance < 20:
        return "0-20m"
    if distance < 40:
        return "20-40m"
    if distance < 60:
        return "40-60m"
    return "60m+"


def speed_bin(speed):
    if not np.isfinite(speed):
        return "unknown"
    if speed < 1:
        return "0-1mps"
    if speed < 5:
        return "1-5mps"
    return "5mps+"


def match_sample(gt_objects, prediction, score_threshold, center_threshold):
    boxes = prediction["boxes_3d"].tensor.detach().cpu().numpy()
    scores = prediction["scores_3d"].detach().cpu().numpy()
    labels = prediction["labels_3d"].detach().cpu().numpy()
    if not (len(boxes) == len(scores) == len(labels)):
        raise ValueError("Prediction boxes, scores and labels have different lengths")

    gt_by_class = defaultdict(list)
    for gi, gt in enumerate(gt_objects):
        gt_by_class[gt["class_name"]].append(gi)

    # Match highest-confidence detections first, separately for each class.
    pred_indices = np.flatnonzero(scores >= score_threshold)
    pred_indices = pred_indices[np.argsort(-scores[pred_indices], kind="stable")]
    matched_gt = set()
    gt_matches = {}
    tp = fp = 0
    for pi in pred_indices:
        label = int(labels[pi])
        if label < 0 or label >= len(CLASSES):
            fp += 1
            continue
        class_name = CLASSES[label]
        candidates = [gi for gi in gt_by_class[class_name] if gi not in matched_gt]
        if not candidates:
            fp += 1
            continue
        dists = [float(np.linalg.norm(boxes[pi, :2] -
                                      [gt_objects[gi]["x"], gt_objects[gi]["y"]]))
                 for gi in candidates]
        best_pos = int(np.argmin(dists))
        if dists[best_pos] <= center_threshold:
            gi = candidates[best_pos]
            matched_gt.add(gi)
            gt_matches[gi] = (float(scores[pi]), dists[best_pos])
            tp += 1
        else:
            fp += 1
    return gt_matches, tp, fp


def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    with open(args.common_tokens, encoding="utf-8") as stream:
        common_list = json.load(stream)
    common = set(common_list)
    scene_by_token = read_scene_map(args.manifest, common)

    info_data = load(args.info)
    infos = sorted(info_data["infos"], key=lambda item: item["timestamp"])
    infos = [item for item in infos if item["token"] in common]
    tokens = [item["token"] for item in infos]
    if len(tokens) != len(common) or set(tokens) != common:
        raise ValueError("Sorted info records do not exactly cover common tokens")
    print(f"Using {len(tokens)} timestamp-sorted samples")

    gt_by_sample = {item["token"]: get_gt(item) for item in infos}
    all_results = {}
    for condition in CONDITIONS:
        path = os.path.join(args.results_dir, condition, "predictions.pkl")
        results = load(path)
        if len(results) != len(tokens):
            raise ValueError(f"{condition}: {len(results)} predictions for {len(tokens)} samples")
        all_results[condition] = results
        print(f"Loaded {condition}: {len(results)} predictions")

    object_rows = []
    summary_acc = defaultdict(lambda: {"gt": 0, "tp": 0, "fp": 0,
                                      "center_error_sum": 0.0, "center_error_n": 0,
                                      "scenes": set()})
    for condition, predictions in all_results.items():
        for sample_idx, (token, pred) in enumerate(zip(tokens, predictions)):
            gt_objects = gt_by_sample[token]
            for score_threshold in args.score_thresholds:
                for center_threshold in args.center_thresholds:
                    matches, tp, fp = match_sample(
                        gt_objects, pred, score_threshold, center_threshold)
                    # Aggregate false positives by class, sample, and threshold.
                    pred_scores = pred["scores_3d"].detach().cpu().numpy()
                    pred_labels = pred["labels_3d"].detach().cpu().numpy()
                    kept = np.flatnonzero(pred_scores >= score_threshold)
                    matched_count = len(matches)
                    # tp/fp above are sample-level; detailed subgroup FP is not
                    # attributed to GT speed/distance because FP has no GT object.
                    for class_name in CLASSES:
                        key = (condition, score_threshold, center_threshold,
                               class_name, "ALL", "ALL")
                        acc = summary_acc[key]
                        gt_indices = [i for i, g in enumerate(gt_objects)
                                      if g["class_name"] == class_name]
                        acc["gt"] += len(gt_indices)
                        class_tps = 0
                        for gi in gt_indices:
                            gt = gt_objects[gi]
                            hit = gi in matches
                            error = matches[gi][1] if hit else ""
                            object_rows.append({
                                "condition": condition,
                                "sample_index": sample_idx,
                                "sample_token": token,
                                "scene_token": scene_by_token[token],
                                "score_threshold": score_threshold,
                                "center_threshold_m": center_threshold,
                                "gt_index": gt["gt_index"],
                                "class_name": class_name,
                                "distance_m": gt["distance_m"],
                                "distance_bin": dist_bin(gt["distance_m"]),
                                "speed_mps": gt["speed_mps"],
                                "speed_bin": speed_bin(gt["speed_mps"]),
                                "matched": int(hit),
                                "matched_prediction_score": matches[gi][0] if hit else "",
                                "center_error_m": error,
                            })
                            if hit:
                                class_tps += 1
                                acc["center_error_sum"] += matches[gi][1]
                                acc["center_error_n"] += 1
                        acc["tp"] += class_tps
                        acc["scenes"].add(scene_by_token[token])

                    # Grouped recall rows: each GT contributes to its own bins.
                    for gi, gt in enumerate(gt_objects):
                        speed_groups = ["ALL"]
                        if np.isfinite(gt["speed_mps"]):
                            speed_groups.append(speed_bin(gt["speed_mps"]))
                        distance_groups = ["ALL", dist_bin(gt["distance_m"])]
                        for sb in speed_groups:
                            for db in distance_groups:
                                # The ALL/ALL cell was accumulated above once
                                # per class, together with its false-positive
                                # accounting. Avoid counting it a second time.
                                if sb == "ALL" and db == "ALL":
                                    continue
                                key = (condition, score_threshold, center_threshold,
                                       gt["class_name"], sb, db)
                                acc = summary_acc[key]
                                acc["gt"] += 1
                                acc["scenes"].add(scene_by_token[token])
                                if gi in matches:
                                    acc["tp"] += 1
                                    acc["center_error_sum"] += matches[gi][1]
                                    acc["center_error_n"] += 1
                    # Record aggregate sample-level FP separately in a special row.
                    summary_acc[(condition, score_threshold, center_threshold,
                                 "__all__", "ALL", "ALL")]["fp"] += fp

    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, "per_gt_matches.csv"), object_rows,
              ["condition", "sample_index", "sample_token", "scene_token",
               "score_threshold", "center_threshold_m", "gt_index", "class_name",
               "distance_m", "distance_bin", "speed_mps", "speed_bin", "matched",
               "matched_prediction_score", "center_error_m"])

    summary_rows = []
    for key, acc in sorted(summary_acc.items(), key=lambda x: str(x[0])):
        condition, score_threshold, center_threshold, class_name, sb, db = key
        if class_name == "__all__":
            continue
        summary_rows.append({
            "condition": condition,
            "score_threshold": score_threshold,
            "center_threshold_m": center_threshold,
            "class_name": class_name,
            "speed_bin": sb,
            "distance_bin": db,
            "gt_objects": acc["gt"],
            "true_positives": acc["tp"],
            "recall": acc["tp"] / acc["gt"] if acc["gt"] else "",
            "mean_matched_center_error_m": (
                acc["center_error_sum"] / acc["center_error_n"]
                if acc["center_error_n"] else ""),
            "scene_count": len(acc["scenes"]),
        })
    write_csv(os.path.join(args.out_dir, "recall_by_group.csv"), summary_rows,
              ["condition", "score_threshold", "center_threshold_m", "class_name",
               "speed_bin", "distance_bin", "gt_objects", "true_positives",
               "recall", "mean_matched_center_error_m", "scene_count"])
    print(f"Wrote {len(object_rows)} GT-condition match rows to {args.out_dir}")
    print("Note: subgroup recall is fixed-score diagnostic; it is not nuScenes AP.")


if __name__ == "__main__":
    main()
