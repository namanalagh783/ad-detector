"""Layer 8 checks: audio extraction against a fixture with a known 5s
tone (not speech -- that's layer 9's concern). Checks the extracted wav
exists, has the right duration, and is properly mono/16kHz.

Runnable directly (`python tests/test_audio.py`) or under pytest.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import subprocess

from src.audio import extract_audio

FIXTURE = str(Path(__file__).parent / "fixtures" / "audio_fixture.mp4")
OUT_DIR = str(Path(__file__).parent / "fixtures" / "_audio_extract_test_out")


def _probe(path: str) -> dict:
    cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def test_extraction_produces_a_file():
    audio_path = extract_audio(FIXTURE, OUT_DIR)
    assert Path(audio_path).exists(), f"extracted audio not found: {audio_path}"


def test_duration_matches_source():
    audio_path = extract_audio(FIXTURE, OUT_DIR)
    info = _probe(audio_path)
    duration = float(info["format"]["duration"])
    assert 4.8 <= duration <= 5.2, f"expected ~5s, got {duration}s"


def test_mono_16khz():
    audio_path = extract_audio(FIXTURE, OUT_DIR)
    info = _probe(audio_path)
    stream = next(s for s in info["streams"] if s["codec_type"] == "audio")
    assert stream["channels"] == 1, f"expected mono, got {stream['channels']} channels"
    assert stream["sample_rate"] == "16000", f"expected 16000Hz, got {stream['sample_rate']}"


CHECKS = [
    ("extraction produces a wav file", test_extraction_produces_a_file),
    ("extracted duration matches source (~5s)", test_duration_matches_source),
    ("extracted audio is mono/16kHz as required by faster-whisper", test_mono_16khz),
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