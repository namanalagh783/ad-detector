"""Layer 1 checks: run_pipeline wires ingest -> detect -> stats correctly,
using fake data at every stage. Doesn't test real video processing --
that's layer 2+.

Runnable directly (`python tests/test_pipeline_skeleton.py`) or under pytest.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import run_pipeline
from src.schema import AdDetectionResult, Kind, Platform


def test_returns_valid_result_with_hints():
    result = run_pipeline(
        "https://fake.example/video",
        platform_hint=Platform.YOUTUBE,
        kind_hint=Kind.VOD,
    )
    assert isinstance(result, AdDetectionResult)


def test_source_echoes_input():
    result = run_pipeline(
        "https://fake.example/video",
        platform_hint=Platform.YOUTUBE,
        kind_hint=Kind.VOD,
    )
    assert result.source.url == "https://fake.example/video"


def test_hints_are_respected():
    result = run_pipeline(
        "https://fake.example/video",
        platform_hint=Platform.YOUTUBE,
        kind_hint=Kind.VOD,
    )
    assert result.source.platform == Platform.YOUTUBE
    assert result.source.kind == Kind.VOD


def test_fake_segment_flows_through():
    result = run_pipeline("https://fake.example/video")
    assert len(result.segments) == 1
    seg = result.segments[0]
    assert seg.ad_type.value == "midroll_sponsor_read"
    assert seg.brand == "Acme VPN"


def test_stats_are_real_timing():
    result = run_pipeline("https://fake.example/video")
    assert result.stats.wall_clock_s >= 0


def test_defaults_without_hints():
    result = run_pipeline("https://fake.example/other")
    assert result.source.platform == Platform.OTHER
    assert result.source.kind == Kind.VOD


CHECKS = [
    ("run_pipeline returns AdDetectionResult", test_returns_valid_result_with_hints),
    ("source.url echoes input", test_source_echoes_input),
    ("platform_hint/kind_hint respected", test_hints_are_respected),
    ("fake segment flows through with correct fields", test_fake_segment_flows_through),
    ("stats.wall_clock_s is real timing (>= 0)", test_stats_are_real_timing),
    ("defaults to OTHER/VOD when no hints given", test_defaults_without_hints),
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