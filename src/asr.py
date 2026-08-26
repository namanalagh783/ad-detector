"""Layer 9: ASR via faster-whisper. This is the PRIMARY detection signal
for long-form video -- sponsor reads are scripted speech, and a 90s
sponsor read where the on-screen visual never changes has literally zero
signal for a frame-only pipeline. Free/local too (no per-call cost),
which matters for staying under the assignment's $10 total-spend cap.

CAVEAT: written and reasoned through carefully, but NOT run end-to-end in
the environment this was built in -- that sandbox has no route to
huggingface.co, where faster-whisper pulls model weights from on first
use. Needs to be verified for real by whoever runs this next. See
tests/test_asr.py, which needs real internet access on first run (to
download weights) and is kept deliberately separate from earlier offline
tests.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import ASRConfig

_model_cache: dict[tuple[str, str, str], "WhisperModel"] = {}  # noqa: F821


@dataclass
class TranscriptSegment:
    start_s: float
    end_s: float
    text: str


def _get_model(cfg: ASRConfig):
    from faster_whisper import WhisperModel  # lazy import: earlier layers never need this

    source = cfg.model_dir or cfg.model_size
    key = (source, cfg.device, cfg.compute_type)
    if key not in _model_cache:
        _model_cache[key] = WhisperModel(source, device=cfg.device, compute_type=cfg.compute_type)
    return _model_cache[key]


def transcribe(audio_path: str, cfg: ASRConfig) -> list[TranscriptSegment]:
    model = _get_model(cfg)
    segments, _info = model.transcribe(audio_path, vad_filter=True)
    return [
        TranscriptSegment(start_s=s.start, end_s=s.end, text=s.text.strip())
        for s in segments
    ]