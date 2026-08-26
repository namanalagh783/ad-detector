"""Layer 6 checks: frame extraction against scenes_fixture.mp4, whose
colors at known times we already know from layer 4's fixture (blue
0-4s, yellow 4-6s, green 6-10s). Checking actual pixel color, not just
"a file exists" -- proves we're extracting the right MOMENT, not just
any frame.

Runnable directly (`python tests/test_frames.py`) or under pytest.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import shutil

import cv2

from src.frames import extract_frames

FIXTURE = str(Path(__file__).parent / "fixtures" / "scenes_fixture.mp4")
OUT_DIR = str(Path(__file__).parent / "fixtures" / "_frame_extract_test_out")

# BGR, since that's what OpenCV reads/writes
EXPECTED_BGR = {
    1.0: (0xaa, 0x44, 0x22),  # blue segment (0x2244aa in RGB -> BGR)
    5.0: (0x00, 0xdd, 0xff),  # yellow segment (0xffdd00 in RGB -> BGR)
    8.0: (0x33, 0x77, 0x11),  # green segment (0x117733 in RGB -> BGR)
}


def _color_close(actual, expected, tol=25) -> bool:
    return all(abs(int(a) - int(e)) <= tol for a, e in zip(actual, expected))


def setup_module():
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)


def test_extracts_a_file_per_timestamp():
    saved = extract_frames(FIXTURE, [1.0, 5.0, 8.0], OUT_DIR)
    assert set(saved.keys()) == {1.0, 5.0, 8.0}
    for ts, path in saved.items():
        assert os.path.exists(path), f"missing frame file for {ts}: {path}"


def test_frame_content_matches_known_color():
    saved = extract_frames(FIXTURE, [1.0, 5.0, 8.0], OUT_DIR)
    for ts, expected in EXPECTED_BGR.items():
        img = cv2.imread(saved[ts])
        assert img is not None, f"could not read saved frame for {ts}"
        h, w = img.shape[:2]
        center_bgr = img[h // 2, w // 2]
        assert _color_close(center_bgr, expected), (
            f"frame at {ts}s: expected ~{expected}, got {tuple(center_bgr)}"
        )


def test_timestamp_past_duration_is_skipped_not_crashed():
    saved = extract_frames(FIXTURE, [1.0, 999.0], OUT_DIR)
    assert 1.0 in saved
    assert 999.0 not in saved  # skipped, no crash


CHECKS = [
    ("extracts one file per requested timestamp", test_extracts_a_file_per_timestamp),
    ("frame content matches known color at each timestamp", test_frame_content_matches_known_color),
    ("timestamp past video duration is skipped, not crashed", test_timestamp_past_duration_is_skipped_not_crashed),
]


def main() -> int:
    setup_module()
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