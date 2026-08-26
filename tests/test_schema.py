"""Layer 0 checks: the contract parses the reference payload and rejects bad segments.

Runnable directly (`python tests/test_schema.py`) or under pytest.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import ValidationError

from src.schema import AdDetectionResult, Segment

EXAMPLE = {
    "source": {
        "url": "https://...",
        "platform": "youtube",
        "kind": "vod",
        "duration_s": 812.4,
        "processed_at": "2026-08-21T10:04:11Z",
    },
    "segments": [
        {
            "id": "seg_01",
            "start_s": 124.0,
            "end_s": 187.5,
            "ad_type": "midroll_sponsor_read",
            "confidence": 0.81,
            "brand": "Acme VPN",
            "description": (
                "Host reads a scripted sponsor spot for a VPN service, cuts to a "
                "screen recording of the app, gives a discount code."
            ),
            "evidence": {
                "frame_timestamps": [125.0, 140.5, 171.0],
                "transcript_span": "…",
                "signals_used": ["asr", "ocr", "vlm_frame", "scene_cut"],
            },
        }
    ],
    "stats": {
        "wall_clock_s": 47.2,
        "estimated_cost_usd": 0.083,
        "frames_sampled": 214,
        "model_calls": 19,
    },
}

BAD_SEGMENT_BASE = {
    "id": "seg_bad",
    "start_s": 100.0,
    "end_s": 200.0,
    "ad_type": "preroll",
    "confidence": 0.5,
    "description": "placeholder",
    "evidence": {"frame_timestamps": [], "signals_used": ["asr"]},
}


def test_example_payload_parses():
    result = AdDetectionResult.model_validate(EXAMPLE)
    seg = result.segments[0]
    assert result.source.platform.value == "youtube"
    assert result.source.kind.value == "vod"
    assert seg.ad_type.value == "midroll_sponsor_read"
    assert seg.confidence == 0.81
    assert [s.value for s in seg.evidence.signals_used] == [
        "asr",
        "ocr",
        "vlm_frame",
        "scene_cut",
    ]
    assert result.stats.model_calls == 19


def test_reversed_span_rejected():
    bad = {**BAD_SEGMENT_BASE, "start_s": 200.0, "end_s": 100.0}
    try:
        Segment.model_validate(bad)
    except ValidationError:
        return
    raise AssertionError("end_s < start_s was accepted")


def test_confidence_out_of_range_rejected():
    bad = {**BAD_SEGMENT_BASE, "confidence": 1.5}
    try:
        Segment.model_validate(bad)
    except ValidationError:
        return
    raise AssertionError("confidence 1.5 was accepted")


CHECKS = [
    ("example JSON parses as AdDetectionResult", test_example_payload_parses),
    ("end_s < start_s raises ValidationError", test_reversed_span_rejected),
    ("confidence = 1.5 raises ValidationError", test_confidence_out_of_range_rejected),
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
