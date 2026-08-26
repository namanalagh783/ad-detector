"""Layer 7 checks: OCR against a control frame ("MAIN CONTENT" -- should
NOT flag as ad) and a bumper frame ("SPONSORED" -- should flag).

Runnable directly (`python tests/test_ocr.py`) or under pytest.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.frames import extract_frames
from src.ocr import ocr_frames

FIXTURES = Path(__file__).parent / "fixtures"
OUT_DIR = str(FIXTURES / "_ocr_test_out")


def test_control_frame_not_flagged():
    frames = extract_frames(str(FIXTURES / "ocr_control.mp4"), [1.0], OUT_DIR)
    hits = ocr_frames(frames)
    assert len(hits) == 1
    assert "CONTENT" in hits[0].text.upper()
    assert hits[0].is_ad_indicator is False


def test_bumper_frame_flagged():
    frames = extract_frames(str(FIXTURES / "ocr_bumper.mp4"), [1.0], OUT_DIR)
    hits = ocr_frames(frames)
    assert len(hits) == 1
    assert "SPONSOR" in hits[0].text.upper()
    assert hits[0].is_ad_indicator is True


def test_multiple_frames_processed_independently():
    control_frames = extract_frames(str(FIXTURES / "ocr_control.mp4"), [1.0], OUT_DIR + "_c")
    bumper_frames = extract_frames(str(FIXTURES / "ocr_bumper.mp4"), [1.0], OUT_DIR + "_b")
    combined = {**control_frames, 2.0: bumper_frames[1.0]}  # fake a shared dict at different keys
    hits = ocr_frames(combined)
    flagged = {h.timestamp_s: h.is_ad_indicator for h in hits}
    assert flagged[1.0] is False
    assert flagged[2.0] is True


CHECKS = [
    ("control frame text read correctly, NOT flagged as ad", test_control_frame_not_flagged),
    ("bumper frame text read correctly, flagged as ad", test_bumper_frame_flagged),
    ("mixed frames flagged independently and correctly", test_multiple_frames_processed_independently),
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