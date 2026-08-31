"""Layer 16 checks: the eval CLI plumbing (run_eval + print_report)
against synthetic ground_truth.json + prediction files. Proves the file
loading, video-id matching, and report formatting all work correctly --
ready to point at the real ground_truth.json once it's hand-labeled.

Runnable directly (`python tests/test_eval_run.py`) or under pytest.
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.run_eval import print_report, run_eval

TMP_DIR = None


def setup():
    global TMP_DIR
    TMP_DIR = tempfile.mkdtemp()

    gt = {
        "video_a": {
            "source": {"url": "https://example.com/a", "duration_s": 100.0},
            "segments": [
                {"id": "gt1", "start_s": 10.0, "end_s": 20.0, "ad_type": "midroll_sponsor_read",
                 "confidence": 1.0, "description": "hand-labeled sponsor read"},
            ],
        },
        "video_b": {
            "source": {"url": "https://example.com/b", "duration_s": 60.0},
            "segments": [
                {"id": "gt1", "start_s": 5.0, "end_s": 15.0, "ad_type": "bumper",
                 "confidence": 1.0, "description": "hand-labeled bumper"},
            ],
        },
    }
    with open(os.path.join(TMP_DIR, "ground_truth.json"), "w") as f:
        json.dump(gt, f)

    pred_dir = os.path.join(TMP_DIR, "preds")
    os.makedirs(pred_dir)

    with open(os.path.join(pred_dir, "video_a.json"), "w") as f:
        json.dump({"segments": [
            {"id": "seg_01", "start_s": 10.5, "end_s": 20.0, "ad_type": "midroll_sponsor_read",
             "confidence": 0.8, "description": "predicted"},
        ]}, f)

    with open(os.path.join(pred_dir, "video_b.json"), "w") as f:
        json.dump({"segments": []}, f)

    return os.path.join(TMP_DIR, "ground_truth.json"), pred_dir


def teardown():
    if TMP_DIR:
        shutil.rmtree(TMP_DIR, ignore_errors=True)


def test_run_eval_loads_and_matches_both_videos():
    gt_path, pred_dir = setup()
    try:
        results = run_eval(gt_path, pred_dir)
        assert set(results.keys()) == {"video_a", "video_b"}
    finally:
        teardown()


def test_near_perfect_prediction_scores_well():
    gt_path, pred_dir = setup()
    try:
        results = run_eval(gt_path, pred_dir)
        r = results["video_a"]
        assert r.precision == 1.0
        assert r.recall == 1.0
        assert r.mean_start_error_s == 0.5
        assert r.mean_end_error_s == 0.0
    finally:
        teardown()


def test_missed_video_scores_zero_recall():
    gt_path, pred_dir = setup()
    try:
        results = run_eval(gt_path, pred_dir)
        r = results["video_b"]
        assert r.recall == 0.0
        assert len(r.false_negative_indices) == 1
    finally:
        teardown()


def test_missing_prediction_file_is_skipped_not_crashed():
    gt_path, pred_dir = setup()
    try:
        os.remove(os.path.join(pred_dir, "video_b.json"))
        results = run_eval(gt_path, pred_dir)
        assert set(results.keys()) == {"video_a"}
    finally:
        teardown()


def test_print_report_does_not_crash():
    gt_path, pred_dir = setup()
    try:
        results = run_eval(gt_path, pred_dir)
        print_report(results)
    finally:
        teardown()


CHECKS = [
    ("run_eval loads ground_truth.json and matches predictions by video id", test_run_eval_loads_and_matches_both_videos),
    ("a near-perfect prediction scores correctly (incl. boundary error)", test_near_perfect_prediction_scores_well),
    ("a completely missed video correctly scores zero recall", test_missed_video_scores_zero_recall),
    ("a missing prediction file is skipped with a warning, not a crash", test_missing_prediction_file_is_skipped_not_crashed),
    ("print_report runs without crashing on real EvalResult objects", test_print_report_does_not_crash),
]


def main() -> int:
    failures = 0
    for name, check in CHECKS:
        try:
            check()
        except Exception as exc:  # noqa: BLE001 - report, don't stop
            failures += 1
            print(f"FAIL  {name}\n        {type(exc).__name__}: {exc}")
        else:
            print(f"PASS  {name}")
    print(f"\n{len(CHECKS) - failures}/{len(CHECKS)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())