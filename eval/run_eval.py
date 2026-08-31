"""Layer 16: CLI eval runner. Compares predicted results (from
scripts/run_once.py or the API) against ground_truth.json.

ground_truth.json format: a JSON object keyed by a short video id, each
value shaped like the output contract. Example:

{
  "longform_1": {
    "source": {"url": "...", "duration_s": 812.4, ...},
    "segments": [{"start_s": 124.0, "end_s": 187.5, "ad_type": "midroll_sponsor_read", ...}, ...]
  },
  ...
}

Predictions are expected as one JSON file per video id, named
<video_id>.json, inside --pred's directory -- e.g. run scripts/run_once.py
once per test-set video and save each as data/outputs/<video_id>.json.

Usage:
  python -m eval.run_eval --gt ground_truth.json --pred data/outputs/
"""

from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace

from eval.metrics import evaluate


def _load_segments(path_or_list) -> list:
    """Accepts either a path to a JSON file shaped like the output
    contract, or an already-loaded list of segment dicts."""
    if isinstance(path_or_list, str):
        with open(path_or_list, "r", encoding="utf-8") as f:
            data = json.load(f)
        segments = data.get("segments", data) if isinstance(data, dict) else data
    else:
        segments = path_or_list
    return [SimpleNamespace(**seg) for seg in segments]


def run_eval(gt_path: str, pred_dir: str, iou_threshold: float = 0.5) -> dict:
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    per_video = {}
    for video_id, gt_entry in gt_data.items():
        pred_path = os.path.join(pred_dir, f"{video_id}.json")
        if not os.path.exists(pred_path):
            print(f"WARNING: no prediction file found for '{video_id}' at {pred_path} -- skipping")
            continue

        gt_segments = _load_segments(gt_entry.get("segments", []))
        pred_segments = _load_segments(pred_path)

        per_video[video_id] = evaluate(pred_segments, gt_segments, iou_threshold=iou_threshold)

    return per_video


def _fmt(value) -> str:
    return f"{value:.2f}" if value is not None else "-"


def print_report(per_video: dict) -> None:
    if not per_video:
        print("No videos evaluated -- check --gt and --pred paths.")
        return

    header = (f"{'video':<15} {'precision':>9} {'recall':>7} {'f1':>6} {'mean_iou':>9} "
              f"{'start_err':>10} {'end_err':>9} {'type_acc':>9}")
    print(header)
    print("-" * len(header))
    for video_id, r in per_video.items():
        print(
            f"{video_id:<15} {r.precision:>9.2f} {r.recall:>7.2f} {r.f1:>6.2f} "
            f"{_fmt(r.mean_iou):>9} {_fmt(r.mean_start_error_s):>10} "
            f"{_fmt(r.mean_end_error_s):>9} {_fmt(r.type_accuracy):>9}"
        )

    n = len(per_video)
    avg_precision = sum(r.precision for r in per_video.values()) / n
    avg_recall = sum(r.recall for r in per_video.values()) / n
    avg_f1 = sum(r.f1 for r in per_video.values()) / n
    print("-" * len(header))
    print(f"{'AVERAGE':<15} {avg_precision:>9.2f} {avg_recall:>7.2f} {avg_f1:>6.2f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate pipeline predictions against ground_truth.json")
    parser.add_argument("--gt", required=True, help="Path to ground_truth.json")
    parser.add_argument("--pred", required=True, help="Directory containing <video_id>.json prediction files")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    per_video = run_eval(args.gt, args.pred, args.iou_threshold)
    print_report(per_video)


if __name__ == "__main__":
    main()