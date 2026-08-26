"""Layer 9 manual check: real transcription via faster-whisper.

THIS NEEDS REAL INTERNET ACCESS on first run (to download model weights
from huggingface.co) and a real spoken-word audio fixture, which you need
to generate yourself first -- see instructions above test_tts_fixture().

Was NOT run successfully in the environment this code was written in
(sandboxed, no route to huggingface.co). Run this yourself and report
back what happens.

Run: python tests/test_asr.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.asr import transcribe
from src.audio import extract_audio
from src.config import ASRConfig

TTS_FIXTURE = str(Path(__file__).parent / "fixtures" / "speech_fixture.wav")
REAL_SHORT = str(Path(__file__).parent.parent / "data" / "videos" / "Ve0zdhTQA4U.mp4")


def test_tts_fixture_transcribes_plausibly():
    """Loose check on purpose -- TTS-through-ASR isn't going to be exact
    word-for-word. We're checking a couple of common, clearly-spoken
    words show up somewhere, and that segment timing is sane -- not
    exact wording or exact boundaries."""
    segments = transcribe(TTS_FIXTURE, ASRConfig())
    assert len(segments) > 0, "got zero transcript segments -- something's wrong"

    full_text = " ".join(s.text for s in segments).lower()
    print(f"    full transcript: {full_text!r}")

    found_any = any(kw in full_text for kw in ["sponsor", "code", "off"])
    assert found_any, f"none of the expected keywords found in: {full_text!r}"

    for s in segments:
        assert s.end_s >= s.start_s, f"segment has end before start: {s}"


def test_real_downloaded_short_transcribes_without_crashing():
    """No ground truth to check against here -- this is a smoke test plus
    a chance to eyeball real output against a real assignment video. Skips
    cleanly if you haven't run the layer 3 YouTube test yet (no file to
    transcribe)."""
    if not Path(REAL_SHORT).exists():
        print(f"    SKIPPED: {REAL_SHORT} not found (run test_ingest_youtube.py first)")
        return

    audio_path = extract_audio(REAL_SHORT, "data/audio")
    segments = transcribe(audio_path, ASRConfig())
    print(f"    {len(segments)} segments transcribed from the real Short:")
    for s in segments:
        print(f"      {s.start_s:.1f}-{s.end_s:.1f}: {s.text}")


CHECKS = [
    ("TTS fixture transcribes with plausible content/timing", test_tts_fixture_transcribes_plausibly),
    ("real downloaded Short transcribes without crashing (eyeball the output)", test_real_downloaded_short_transcribes_without_crashing),
]


def main() -> int:
    print("NOTE: this test needs real internet access on first run (model download).\n")
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