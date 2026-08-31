# EVAL.md — Ad Segment Detection Pipeline

## Ground truth methodology

*(TODO — write this once ground_truth.json is actually hand-labeled.
Should cover: how each of the 5 videos was watched/scrubbed, how the
DESIGN.md ambiguity rulings were applied to real ambiguous moments
encountered while labeling, and how confident you are in your own
labels — e.g. did you re-watch any segment twice, did boundaries feel
uncertain anywhere.)*

`ground_truth.json` format (already implemented and tested in
`eval/run_eval.py`): a JSON object keyed by a short video id, each value
shaped like the pipeline's own output contract (`source` + `segments`),
so the same `Segment` schema is used for both predictions and ground
truth — no separate labeling format to keep in sync.

## Metrics

Defined and implemented in `eval/metrics.py`, unit-tested against
hand-computed synthetic scenarios (`tests/test_eval_metrics.py`, 12/12
passing) before ever being pointed at real predictions.

- **Temporal IoU** — intersection over union of two time spans:
  `intersection / ((end_a - start_a) + (end_b - start_b) - intersection)`.
  0.0 for non-overlapping segments, including segments that merely touch
  at an endpoint.
- **Matching** — greedy, by highest IoU first: among all (ground-truth,
  predicted) pairs at or above `iou_threshold` (default 0.5), the
  highest-IoU pair is assigned first, then the next highest among what's
  left, and so on. Each ground-truth segment and each predicted segment
  can be used in at most one match. **This is a real simplification, not
  a claim of optimality** — a true optimal assignment would need the
  Hungarian algorithm. Defensible given the small number of segments per
  video expected here, not free rigor.
- **Precision / Recall / F1** — standard definitions over matched pairs
  at the IoU threshold. Edge case convention: an empty prediction list
  against a non-empty ground truth scores precision=1.0 (vacuously, no
  wrong predictions were made) but recall=0.0 (everything real was
  missed) — these are reported separately specifically so a review of
  the numbers can't hide "found nothing" behind a misleadingly high
  single score.
- **Boundary error** — mean absolute difference (start and end reported
  separately, not averaged together) in seconds, computed only over
  matched pairs. This is the number the assignment brief specifically
  calls out as more informative than raw accuracy, since it shows *how
  wrong* a correct-ish detection is, not just whether it counts as a hit.
- **Type accuracy** — among matched pairs (correct in time), the
  fraction that also got the right `ad_type`. Kept separate from
  precision/recall on purpose: a segment can be found in the right place
  but mislabeled (this actually happened during development — see
  DESIGN.md Section 4's `bumper` vs `product_placement` bug), and that's
  a different failure mode from missing it entirely.

## Per-video results

*(TODO — blocked on ground_truth.json + running the real pipeline
against all 5 required test-set videos. `eval/run_eval.py` is built and
tested against synthetic data (`tests/test_eval_run.py`, 5/5 passing);
this table is just `python -m eval.run_eval --gt ground_truth.json --pred data/outputs/`
away once those two things exist.)*

| Video | Precision | Recall | F1 | Mean IoU | Start err (s) | End err (s) | Type acc |
|---|---|---|---|---|---|---|---|
| long-form (ujFWRFYLGjY) | - | - | - | - | - | - | - |
| short (Ve0zdhTQA4U) | - | - | - | - | - | - | - |
| live (s0LLVQeMmtU) | - | - | - | - | - | - | - |
| reel 1 (DJl6-v8oufg) | - | - | - | - | - | - | - |
| reel 2 (Db78RoIuOyl) | - | - | - | - | - | - | - |

## Failure cases (three, with root cause)

*(TODO — pick the three most informative real failures once the table
above exists. The brief is explicit that this section matters more than
the results table itself. One strong candidate already surfaced during
development, worth confirming against real ground truth once labeled:)*

**Candidate failure case 1 (from development, not yet a scored eval
failure):** in a controlled test with a known sponsor segment spanning a
bumper card ("SPONSORED") followed immediately by a plain brand-name
visual card ("ACME VPN," no other text), the real (non-mocked) pipeline
correctly detected and classified the bumper portion, but did not extend
the segment to include the brand-name-only portion. Root cause: neither
OCR (the text "ACME VPN" doesn't match any ad-indicator keyword — it's
just a brand name, not promotional language) nor the VLM (Gemini did not
judge a plain brand card as clearly ad-visual on its own, without
additional context) flagged those frames as ad-related. This is arguably
a *correct*, defensible judgment by both signals given what's actually
on screen — but it means the detected segment under-covers the true ad.
Whether this counts as a real failure or a reasonable conservative
judgment call is exactly the kind of case that belongs in this section
once there's real ground truth to compare against.

*(Two more failure cases needed once real eval results exist.)*