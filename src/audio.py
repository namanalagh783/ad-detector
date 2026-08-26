"""Layer 8: audio extraction. Pulls the audio track out of a video into a
mono 16kHz wav -- the exact format faster-whisper (layer 9) expects."""

from __future__ import annotations

import os
import subprocess


def extract_audio(video_path: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join(out_dir, f"{base}.wav")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ac", "1", "-ar", "16000", "-vn",
        audio_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return audio_path