#!/usr/bin/env python3
"""Evaluate a saved tools/test.py prediction pickle without rerunning inference."""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from torchpack.utils.config import configs
from mmengine.config import Config
from mmengine.fileio import load
from mmengine.registry import DefaultScope
from mmdet3d.datasets import build_dataset
from mmdet3d.utils import recursive_eval
from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--common-tokens", required=True)
    parser.add_argument("--jsonfile-prefix", required=True)
    args = parser.parse_args()

    with open(args.common_tokens, encoding="utf-8") as stream:
        common_tokens = set(json.load(stream))

    # The official evaluator normally loads GT for all of val. Pilot
    # predictions contain only the common subset, so filter GT to the same
    # token cohort before DetectionEval checks the sample sets.
    from nuscenes.eval.detection import evaluate as detection_evaluate
    original_load_gt = detection_evaluate.load_gt

    def load_common_gt(*load_args, **load_kwargs):
        gt_boxes = original_load_gt(*load_args, **load_kwargs)
        gt_boxes.boxes = {
            token: gt_boxes.boxes[token]
            for token in sorted(common_tokens)
            if token in gt_boxes.boxes
        }
        missing_tokens = common_tokens.difference(gt_boxes.boxes)
        if missing_tokens:
            raise ValueError(
                f"{len(missing_tokens)} common sample tokens are absent from "
                "the selected nuScenes evaluation split."
            )
        return gt_boxes

    detection_evaluate.load_gt = load_common_gt

    original_load = NuScenesDataset.load_annotations

    def filtered_load(self, ann_file):
        infos = original_load(self, ann_file)
        return [info for info in infos if info["token"] in common_tokens]

    NuScenesDataset.load_annotations = filtered_load
    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)
    cfg.data.test.test_mode = True
    DefaultScope.get_instance("bevfusion-temporal-eval", scope_name="mmdet")
    dataset = build_dataset(cfg.data.test)
    predictions = load(args.predictions)
    os.makedirs(os.path.dirname(os.path.abspath(args.jsonfile_prefix)), exist_ok=True)
    metrics = dataset.evaluate(
        predictions,
        metric=["bbox"],
        jsonfile_prefix=args.jsonfile_prefix,
    )
    print(metrics)


if __name__ == "__main__":
    main()

