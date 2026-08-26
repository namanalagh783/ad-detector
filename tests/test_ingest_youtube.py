"""Layer 3 manual check: real YouTube download via yt-dlp.

THIS TEST NEEDS REAL INTERNET ACCESS TO YOUTUBE.COM.
It was NOT run successfully in the environment this code was written in
(sandboxed, no route to youtube.com -- confirmed via SSL error). Run this
yourself and report back what happens.

Uses the assignment's own YouTube Short as the test subject -- it's small
and fast to download, and it's literally part of your required test set,
so this doubles as real progress on the assignment.

Run: python tests/test_ingest_youtube.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os

from src.ingest import ingest_source
from src.schema import Kind, Platform

SHORT_URL = "https://www.youtube.com/shorts/LeoaKkKFRUw"


def test_download_real_short():
    asset = ingest_source(SHORT_URL, download_dir="data/videos")

    assert asset.platform == Platform.YOUTUBE
    assert asset.kind == Kind.SHORT  # from URL shape (/shorts/)
    assert os.path.exists(asset.local_path), f"downloaded file missing: {asset.local_path}"
    # brief says Shorts run 5-90s -- sanity check we actually got a short clip
    assert 1.0 <= asset.duration_s <= 120.0, f"unexpected duration: {asset.duration_s}"
    assert asset.fps > 0
    assert asset.width > 0 and asset.height > 0

    print(f"    downloaded to: {asset.local_path}")
    print(f"    duration_s={asset.duration_s:.1f}  fps={asset.fps:.1f}  "
          f"resolution={asset.width}x{asset.height}")


CHECKS = [
    ("real download of assignment's YouTube Short succeeds with sane metadata", test_download_real_short),
]


def main() -> int:
    print("NOTE: this test needs real internet access and will actually download a video.\n")
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