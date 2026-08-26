"""Layer 11 manual check: real VLM classification via Gemini.

NEEDS a GEMINI_API_KEY, either in a .env file (loaded automatically via
load_env()) or set as a real environment variable. Reuses the OCR
fixtures from layer 7 (ocr_control.mp4 / ocr_bumper.mp4) -- a nice
cross-check: does the VISUAL signal agree with the TEXT signal on the
same frames?

Free tier -- effectively no cost for these 4-5 calls.

Run: python tests/test_vlm.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env import load_env
load_env()

import os

from src.config import VLMConfig
from src.frames import extract_frames
from src.vlm import classify_candidates

FIXTURES = Path(__file__).parent / "fixtures"
OUT_DIR = str(FIXTURES / "_vlm_test_out")


def test_api_key_present():
    assert os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"), (
        "No GEMINI_API_KEY found. Either put GEMINI_API_KEY=... in a .env "
        "file at the repo root, or run $env:GEMINI_API_KEY = \"...\" first."
    )


def test_bumper_frame_classified_as_ad():
    frames = extract_frames(str(FIXTURES / "ocr_bumper.mp4"), [1.0], OUT_DIR)
    results = classify_candidates(frames, VLMConfig())
    assert len(results) == 1
    r = results[0]
    assert r.error is None, f"VLM call failed: {r.error}"
    print(f"    bumper frame -> is_ad_visual={r.is_ad_visual}, brand={r.brand!r}, "
          f"confidence={r.confidence}, description={r.description!r}")
    assert r.is_ad_visual is True, "expected the SPONSORED bumper to be classified as an ad visual"


def test_control_frame_classified_as_not_ad():
    frames = extract_frames(str(FIXTURES / "ocr_control.mp4"), [1.0], OUT_DIR)
    results = classify_candidates(frames, VLMConfig())
    assert len(results) == 1
    r = results[0]
    assert r.error is None, f"VLM call failed: {r.error}"
    print(f"    control frame -> is_ad_visual={r.is_ad_visual}, brand={r.brand!r}, "
          f"confidence={r.confidence}, description={r.description!r}")
    assert r.is_ad_visual is False, "expected plain 'MAIN CONTENT' frame to NOT be classified as an ad visual"


def test_cost_cap_is_respected():
    frames = extract_frames(str(FIXTURES / "ocr_control.mp4"), [0.5, 1.0, 1.5], OUT_DIR)
    cfg = VLMConfig(max_calls_per_video=1)
    results = classify_candidates(frames, cfg)
    assert len(results) == 1, f"expected exactly 1 result under the cap, got {len(results)}"


CHECKS = [
    ("GEMINI_API_KEY is available", test_api_key_present),
    ("bumper frame (SPONSORED) classified as ad visual", test_bumper_frame_classified_as_ad),
    ("control frame (MAIN CONTENT) classified as NOT ad visual", test_control_frame_classified_as_not_ad),
    ("max_calls_per_video cap is respected", test_cost_cap_is_respected),
]


def main() -> int:
    print("NOTE: this test needs a Gemini API key and makes real (free-tier) API calls.\n")
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