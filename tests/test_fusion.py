"""Layer 12 checks: fusion, tested with REAL confirmed outputs from
layers 4 (scene cuts), 7 (OCR), 9 (ASR), and 11 (VLM) -- not hand-typed
fake data. Specifically:
  - the exact real transcript text your ASR produced in layer 9
  - the exact OCR behavior confirmed in layer 7 (SPONSORED flagged,
    MAIN CONTENT not flagged)
  - the exact VLM judgments Gemini returned in layer 11 for those same
    two frames
  - scene cuts matching the original synthetic long-form fixture design
    (bumper 5.0-6.4s, sponsor visual 6.4-11.0s)

Runnable directly (`python tests/test_fusion.py`) or under pytest.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.asr import TranscriptSegment
from src.fusion import fuse
from src.ocr import OcrHit
from src.vlm import VlmFrameResult


def _composite_scenario():
    transcript = [
        TranscriptSegment(5.0, 8.0, "this video is sponsored by acme vpn."),
        TranscriptSegment(8.0, 11.0, "use promo code save10 at checkout for 20% off."),
    ]
    ocr_hits = [
        OcrHit(timestamp_s=5.3, text="SPONSORED", is_ad_indicator=True),
        OcrHit(timestamp_s=0.5, text="MAIN CONTENT", is_ad_indicator=False),
    ]
    vlm_results = [
        VlmFrameResult(
            timestamp_s=5.3, is_ad_visual=True, brand=None, confidence=0.85,
            description="A bright yellow screen displaying the word SPONSORED in centered black text.",
        ),
        VlmFrameResult(
            timestamp_s=0.5, is_ad_visual=False, brand=None, confidence=0.75,
            description="A solid blue screen features white text reading MAIN CONTENT.",
        ),
    ]
    scene_cuts = [5.0, 6.4, 11.0]
    sample_plan = [0.0, 0.5, 5.0, 5.3, 5.8, 6.3, 6.4, 6.7, 7.2, 7.7, 10.0, 11.0, 11.3, 15.0]
    return transcript, ocr_hits, vlm_results, scene_cuts, sample_plan


def test_produces_exactly_one_merged_segment():
    transcript, ocr_hits, vlm_results, scene_cuts, sample_plan = _composite_scenario()
    segments = fuse(transcript, ocr_hits, vlm_results, scene_cuts, sample_plan)
    assert len(segments) == 1, f"expected 1 merged segment, got {len(segments)}: {segments}"


def test_segment_boundaries_snap_to_scene_cuts():
    transcript, ocr_hits, vlm_results, scene_cuts, sample_plan = _composite_scenario()
    seg = fuse(transcript, ocr_hits, vlm_results, scene_cuts, sample_plan)[0]
    assert seg.start_s == 5.0, f"expected start snapped to 5.0, got {seg.start_s}"
    assert seg.end_s == 11.0, f"expected end snapped to 11.0, got {seg.end_s}"


def test_ad_type_is_sponsor_read_when_transcript_has_launch_and_cta():
    transcript, ocr_hits, vlm_results, scene_cuts, sample_plan = _composite_scenario()
    seg = fuse(transcript, ocr_hits, vlm_results, scene_cuts, sample_plan)[0]
    assert seg.ad_type.value == "midroll_sponsor_read"


def test_brand_extracted_from_real_asr_text():
    transcript, ocr_hits, vlm_results, scene_cuts, sample_plan = _composite_scenario()
    seg = fuse(transcript, ocr_hits, vlm_results, scene_cuts, sample_plan)[0]
    assert seg.brand == "Acme Vpn", f"expected 'Acme Vpn', got {seg.brand!r}"


def test_all_four_signals_recorded_as_evidence():
    transcript, ocr_hits, vlm_results, scene_cuts, sample_plan = _composite_scenario()
    seg = fuse(transcript, ocr_hits, vlm_results, scene_cuts, sample_plan)[0]
    used = {s.value for s in seg.evidence.signals_used}
    assert used == {"asr", "ocr", "vlm_frame", "scene_cut"}, f"got {used}"


def test_transcript_span_preserved_as_evidence():
    transcript, ocr_hits, vlm_results, scene_cuts, sample_plan = _composite_scenario()
    seg = fuse(transcript, ocr_hits, vlm_results, scene_cuts, sample_plan)[0]
    assert "sponsored by acme vpn" in seg.evidence.transcript_span
    assert "save10" in seg.evidence.transcript_span


def test_no_signal_produces_no_segments():
    transcript = [TranscriptSegment(0.0, 3.0, "just talking about nothing in particular.")]
    segments = fuse(transcript, [], [], [], [0.0, 1.0, 2.0])
    assert segments == []


CHECKS = [
    ("real signals merge into exactly one segment", test_produces_exactly_one_merged_segment),
    ("segment boundaries snap to real scene cuts (5.0-11.0)", test_segment_boundaries_snap_to_scene_cuts),
    ("ad_type correctly classified as midroll_sponsor_read", test_ad_type_is_sponsor_read_when_transcript_has_launch_and_cta),
    ("brand extracted from real (lowercase) ASR text", test_brand_extracted_from_real_asr_text),
    ("all 4 signal types recorded as evidence", test_all_four_signals_recorded_as_evidence),
    ("real transcript text preserved in evidence.transcript_span", test_transcript_span_preserved_as_evidence),
    ("no signal at all produces no segments", test_no_signal_produces_no_segments),
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