"""Layer 16 checks: evaluation metrics against synthetic scenarios with
hand-computed expected values -- catching arithmetic mistakes matters
more here than almost anywhere else in the project, since these numbers
are what the whole assignment is graded on.

Runnable directly (`python tests/test_eval_metrics.py`) or under pytest.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import evaluate, temporal_iou


@dataclass
class FakeSeg:
    start_s: float
    end_s: float
    ad_type: str = "midroll_sponsor_read"


def test_iou_perfect_overlap():
    a = FakeSeg(10.0, 20.0)
    b = FakeSeg(10.0, 20.0)
    assert temporal_iou(a, b) == 1.0


def test_iou_no_overlap():
    a = FakeSeg(0.0, 5.0)
    b = FakeSeg(10.0, 15.0)
    assert temporal_iou(a, b) == 0.0


def test_iou_hand_computed_partial_overlap():
    a = FakeSeg(10.0, 20.0)
    b = FakeSeg(15.0, 25.0)
    expected = 5.0 / 15.0
    assert abs(temporal_iou(a, b) - expected) < 1e-9


def test_iou_touching_but_not_overlapping():
    a = FakeSeg(0.0, 5.0)
    b = FakeSeg(5.0, 10.0)
    assert temporal_iou(a, b) == 0.0


def test_evaluate_perfect_match():
    gt = [FakeSeg(10.0, 20.0)]
    pred = [FakeSeg(10.0, 20.0)]
    result = evaluate(pred, gt)
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.mean_iou == 1.0
    assert result.mean_start_error_s == 0.0
    assert result.mean_end_error_s == 0.0
    assert len(result.false_positive_indices) == 0
    assert len(result.false_negative_indices) == 0


def test_evaluate_completely_missed():
    gt = [FakeSeg(10.0, 20.0)]
    pred = []
    result = evaluate(pred, gt)
    assert result.recall == 0.0
    assert result.precision == 1.0
    assert result.false_negative_indices == [0]


def test_evaluate_pure_false_positive():
    gt = []
    pred = [FakeSeg(10.0, 20.0)]
    result = evaluate(pred, gt)
    assert result.precision == 0.0
    assert result.recall == 1.0
    assert result.false_positive_indices == [0]


def test_evaluate_below_threshold_counts_as_miss():
    gt = [FakeSeg(0.0, 10.0)]
    pred = [FakeSeg(9.0, 19.0)]
    result = evaluate(pred, gt, iou_threshold=0.5)
    assert len(result.matches) == 0
    assert result.false_negative_indices == [0]
    assert result.false_positive_indices == [0]


def test_evaluate_boundary_error_measured_correctly():
    gt = [FakeSeg(10.0, 20.0)]
    pred = [FakeSeg(11.0, 22.0)]
    result = evaluate(pred, gt)
    assert result.mean_start_error_s == 1.0
    assert result.mean_end_error_s == 2.0


def test_evaluate_multiple_segments_matched_correctly_not_crossed():
    gt = [FakeSeg(0.0, 10.0), FakeSeg(100.0, 110.0)]
    pred = [FakeSeg(101.0, 109.0), FakeSeg(1.0, 9.0)]
    result = evaluate(pred, gt)
    assert len(result.matches) == 2
    matched_pairs = {(m.gt_index, m.pred_index) for m in result.matches}
    assert (0, 1) in matched_pairs
    assert (1, 0) in matched_pairs


def test_evaluate_type_mismatch_detected():
    gt = [FakeSeg(10.0, 20.0, ad_type="midroll_sponsor_read")]
    pred = [FakeSeg(10.0, 20.0, ad_type="bumper")]
    result = evaluate(pred, gt)
    assert len(result.matches) == 1
    assert result.matches[0].type_match is False
    assert result.type_accuracy == 0.0


def test_evaluate_both_empty_is_perfect():
    result = evaluate([], [])
    assert result.precision == 1.0
    assert result.recall == 1.0


CHECKS = [
    ("IoU: perfect overlap = 1.0", test_iou_perfect_overlap),
    ("IoU: no overlap = 0.0", test_iou_no_overlap),
    ("IoU: hand-computed partial overlap matches exactly", test_iou_hand_computed_partial_overlap),
    ("IoU: touching but not overlapping = 0.0", test_iou_touching_but_not_overlapping),
    ("evaluate: perfect match -> precision/recall/IoU all 1.0, zero boundary error", test_evaluate_perfect_match),
    ("evaluate: completely missed ground truth -> recall 0, precision vacuously 1", test_evaluate_completely_missed),
    ("evaluate: pure false positive -> precision 0, recall vacuously 1", test_evaluate_pure_false_positive),
    ("evaluate: below-threshold overlap counts as a miss, not a match", test_evaluate_below_threshold_counts_as_miss),
    ("evaluate: boundary error measured correctly on both ends", test_evaluate_boundary_error_measured_correctly),
    ("evaluate: multiple segments matched correctly, not cross-paired", test_evaluate_multiple_segments_matched_correctly_not_crossed),
    ("evaluate: ad_type mismatch detected on a timing-correct match", test_evaluate_type_mismatch_detected),
    ("evaluate: both empty -> perfect score", test_evaluate_both_empty_is_perfect),
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