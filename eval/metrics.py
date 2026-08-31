"""Layer 16: evaluation metrics. Segment-level temporal IoU and boundary
error, per the assignment's explicit ask (Section 8): "segment level IoU
and boundary error are more informative than raw accuracy."

Matching is greedy-by-IoU: among all (ground-truth, predicted) pairs at
or above iou_threshold, assign the highest-IoU pair first, then the next
highest among what's left, and so on -- each ground-truth segment and
each predicted segment can be used in at most one match. This is a
reasonable, defensible choice given we're talking about a handful of
segments per video, not a claim that it's globally optimal (a true
optimal assignment would need the Hungarian algorithm). Worth saying so
plainly rather than dressing this up as more rigorous than it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol


class _HasSpan(Protocol):
    start_s: float
    end_s: float


def temporal_iou(a: _HasSpan, b: _HasSpan) -> float:
    """Intersection over union of two time spans. 0.0 if they don't
    overlap at all."""
    intersection = max(0.0, min(a.end_s, b.end_s) - max(a.start_s, b.start_s))
    union = (a.end_s - a.start_s) + (b.end_s - b.start_s) - intersection
    return intersection / union if union > 0 else 0.0


@dataclass
class Match:
    gt_index: int
    pred_index: int
    iou: float
    start_error_s: float
    end_error_s: float
    type_match: bool


@dataclass
class EvalResult:
    matches: list = field(default_factory=list)
    false_positive_indices: list = field(default_factory=list)  # predicted, unmatched
    false_negative_indices: list = field(default_factory=list)  # ground truth, unmatched
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    mean_iou: Optional[float] = None
    mean_start_error_s: Optional[float] = None
    mean_end_error_s: Optional[float] = None
    type_accuracy: Optional[float] = None  # among matches, fraction with matching ad_type


def evaluate(predicted: list, ground_truth: list, iou_threshold: float = 0.5) -> EvalResult:
    """predicted / ground_truth: lists of objects with .start_s, .end_s,
    and (optionally) .ad_type -- schema.Segment works directly, so does
    any object with those attributes."""
    candidate_pairs = []
    for gi, gt in enumerate(ground_truth):
        for pi, pred in enumerate(predicted):
            iou = temporal_iou(gt, pred)
            if iou >= iou_threshold:
                candidate_pairs.append((iou, gi, pi))

    candidate_pairs.sort(key=lambda x: x[0], reverse=True)

    matched_gt = set()
    matched_pred = set()
    matches = []
    for iou, gi, pi in candidate_pairs:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        gt, pred = ground_truth[gi], predicted[pi]
        matches.append(Match(
            gt_index=gi, pred_index=pi, iou=iou,
            start_error_s=abs(gt.start_s - pred.start_s),
            end_error_s=abs(gt.end_s - pred.end_s),
            type_match=(getattr(gt, "ad_type", None) == getattr(pred, "ad_type", None)),
        ))

    fp = [pi for pi in range(len(predicted)) if pi not in matched_pred]
    fn = [gi for gi in range(len(ground_truth)) if gi not in matched_gt]

    tp = len(matches)
    # Convention for the empty-list edge cases: no predictions means no
    # false positives (vacuously precision=1.0), but if ground truth was
    # non-empty, recall is still 0 (everything real got missed).
    precision = tp / len(predicted) if predicted else 1.0
    recall = tp / len(ground_truth) if ground_truth else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    mean_iou = sum(m.iou for m in matches) / tp if tp else None
    mean_start_error = sum(m.start_error_s for m in matches) / tp if tp else None
    mean_end_error = sum(m.end_error_s for m in matches) / tp if tp else None
    type_accuracy = sum(1 for m in matches if m.type_match) / tp if tp else None

    return EvalResult(
        matches=matches,
        false_positive_indices=fp,
        false_negative_indices=fn,
        precision=round(precision, 3),
        recall=round(recall, 3),
        f1=round(f1, 3),
        mean_iou=round(mean_iou, 3) if mean_iou is not None else None,
        mean_start_error_s=round(mean_start_error, 3) if mean_start_error is not None else None,
        mean_end_error_s=round(mean_end_error, 3) if mean_end_error is not None else None,
        type_accuracy=round(type_accuracy, 3) if type_accuracy is not None else None,
    )