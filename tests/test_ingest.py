"""Layer 2 checks: platform/kind detection (pure, no I/O) and real
ffprobe-based ingestion against a local fixture file.

Runnable directly (`python tests/test_ingest.py`) or under pytest.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingest import MediaAsset, detect_kind_from_url, detect_platform, ingest_source
from src.schema import Kind, Platform

FIXTURE = str(Path(__file__).parent / "fixtures" / "sample.mp4")


def test_detect_platform_youtube():
    assert detect_platform("https://www.youtube.com/watch?v=abc123") == Platform.YOUTUBE
    assert detect_platform("https://youtu.be/abc123") == Platform.YOUTUBE


def test_detect_platform_instagram():
    assert detect_platform("https://www.instagram.com/reel/xyz") == Platform.INSTAGRAM


def test_detect_platform_local_file():
    assert detect_platform(FIXTURE) == Platform.FILE


def test_detect_platform_other_url():
    assert detect_platform("https://example.com/video.mp4") == Platform.OTHER


def test_detect_kind_from_url_shorts():
    assert detect_kind_from_url("https://www.youtube.com/shorts/abc123") == Kind.SHORT


def test_detect_kind_from_url_reel():
    assert detect_kind_from_url("https://www.instagram.com/reel/xyz") == Kind.SHORT


def test_detect_kind_from_url_unknown():
    assert detect_kind_from_url("https://www.youtube.com/watch?v=abc123") is None


def test_ingest_local_file_real_metadata():
    asset = ingest_source(FIXTURE)
    assert isinstance(asset, MediaAsset)
    assert asset.platform == Platform.FILE
    assert asset.kind == Kind.VOD
    assert 5.9 <= asset.duration_s <= 6.1
    assert asset.fps > 0
    assert asset.width == 640
    assert asset.height == 360


def test_ingest_local_file_with_kind_hint():
    asset = ingest_source(FIXTURE, kind_hint=Kind.SHORT)
    assert asset.kind == Kind.SHORT


def test_ingest_local_file_as_instagram():
    asset = ingest_source(FIXTURE, platform_hint=Platform.INSTAGRAM, kind_hint=Kind.SHORT)
    assert asset.platform == Platform.INSTAGRAM
    assert asset.kind == Kind.SHORT


def test_ingest_youtube_raises_not_implemented():
    try:
        ingest_source("https://www.youtube.com/watch?v=abc123")
    except NotImplementedError:
        return
    raise AssertionError("expected NotImplementedError for YouTube URL at this layer")


def test_ingest_missing_file_raises():
    try:
        ingest_source("tests/fixtures/does_not_exist.mp4")
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError for missing local file")


CHECKS = [
    ("detect_platform: youtube.com / youtu.be -> YOUTUBE", test_detect_platform_youtube),
    ("detect_platform: instagram.com -> INSTAGRAM", test_detect_platform_instagram),
    ("detect_platform: local path -> FILE", test_detect_platform_local_file),
    ("detect_platform: other URL -> OTHER", test_detect_platform_other_url),
    ("detect_kind_from_url: /shorts/ -> SHORT", test_detect_kind_from_url_shorts),
    ("detect_kind_from_url: instagram.com/reel -> SHORT", test_detect_kind_from_url_reel),
    ("detect_kind_from_url: plain watch URL -> None (ambiguous)", test_detect_kind_from_url_unknown),
    ("ingest_source: real ffprobe metadata from local file", test_ingest_local_file_real_metadata),
    ("ingest_source: kind_hint overrides default", test_ingest_local_file_with_kind_hint),
    ("ingest_source: local file as manually-acquired Reel", test_ingest_local_file_as_instagram),
    ("ingest_source: YouTube URL raises NotImplementedError (expected, layer 3)", test_ingest_youtube_raises_not_implemented),
    ("ingest_source: missing local file raises FileNotFoundError", test_ingest_missing_file_raises),
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