"""Layer 13 checks: full pipeline orchestration. transcribe() and
classify_candidates() are mocked (see module docstring in src/pipeline.py
for why -- network/API-key constraints in this environment) -- everything
else (ingestion, scene cuts, sampling, frame extraction, OCR, fusion)
runs for REAL against tests/fixtures/pipeline_fixture.mp4.

Runnable directly (`python tests/test_pipeline_full.py`) or under pytest.
"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.asr import TranscriptSegment
from src.pipeline import run_pipeline
from src.schema import AdDetectionResult, Kind, Platform
from src.vlm import VlmFrameResult

FIXTURE = str(Path(__file__).parent / "fixtures" / "pipeline_fixture.mp4")

FAKE_TRANSCRIPT = [
    TranscriptSegment(0.0, 3.8, "welcome back to the show today."),
    TranscriptSegment(4.0, 7.0, "this video is sponsored by acme vpn."),
    TranscriptSegment(7.0, 9.0, "use promo code save10 for 20% off."),
    TranscriptSegment(9.0, 12.8, "anyway let's get back to it."),
]


def _fake_classify_candidates(image_paths, cfg, client=None):
    results = []
    for ts in image_paths:
        if 4.0 <= ts <= 9.0:
            results.append(VlmFrameResult(
                timestamp_s=ts, is_ad_visual=True, brand="Acme Vpn" if ts >= 5.4 else None,
                description="sponsor bumper or product visual", confidence=0.85,
            ))
        else:
            results.append(VlmFrameResult(
                timestamp_s=ts, is_ad_visual=False, brand=None,
                description="ordinary content", confidence=0.7,
            ))
    return results


def _run():
    with patch("src.pipeline.transcribe", return_value=FAKE_TRANSCRIPT), \
         patch("src.pipeline.classify_candidates", side_effect=_fake_classify_candidates):
        return run_pipeline(FIXTURE, platform_hint=Platform.FILE, kind_hint=Kind.VOD)


def test_full_pipeline_end_to_end():
    result = _run()
    assert isinstance(result, AdDetectionResult)


def test_source_metadata_is_real():
    result = _run()
    assert result.source.platform == Platform.FILE
    assert result.source.kind == Kind.VOD
    assert 12.5 <= result.source.duration_s <= 13.5


def test_produces_the_expected_sponsor_segment():
    result = _run()
    assert len(result.segments) == 1, f"expected 1 segment, got {len(result.segments)}: {result.segments}"
    seg = result.segments[0]
    assert seg.ad_type.value == "midroll_sponsor_read"
    assert seg.brand == "Acme Vpn"
    assert 3.5 <= seg.start_s <= 4.5
    assert 8.5 <= seg.end_s <= 9.5


def test_real_ocr_and_asr_and_vlm_signals_present():
    result = _run()
    seg = result.segments[0]
    used = {s.value for s in seg.evidence.signals_used}
    assert "ocr" in used, f"expected real OCR to contribute, signals were: {used}"
    assert "asr" in used
    assert "vlm_frame" in used


def test_stats_reflect_real_work():
    result = _run()
    assert result.stats.frames_sampled > 0
    assert result.stats.wall_clock_s > 0
    assert result.stats.model_calls > 0


CHECKS = [
    ("full pipeline runs end-to-end and returns valid result", test_full_pipeline_end_to_end),
    ("source metadata is real (ffprobe-derived)", test_source_metadata_is_real),
    ("produces the expected sponsor segment with correct type/brand/timing", test_produces_the_expected_sponsor_segment),
    ("real OCR + ASR + VLM signals all present in evidence", test_real_ocr_and_asr_and_vlm_signals_present),
    ("stats reflect real work done", test_stats_reflect_real_work),
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