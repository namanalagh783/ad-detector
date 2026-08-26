"""Layer 6: frame extraction at specific timestamps.

Uses one cv2.VideoCapture handle, seeked repeatedly, rather than spawning
an ffmpeg process per frame -- matters once a sample plan has hundreds of
timestamps (see SamplingConfig.max_frames_per_video).
"""

from __future__ import annotations

import os

import cv2


def extract_frames(video_path: str, timestamps: list[float], out_dir: str) -> dict[float, str]:
    """Returns {timestamp: saved_frame_path}. Skips a timestamp if the
    seek/read fails (e.g. past EOF due to float rounding) rather than
    raising -- a missing frame should degrade sampling density, not crash
    the whole pipeline run."""
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    saved: dict[float, str] = {}
    try:
        for ts in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            path = os.path.join(out_dir, f"frame_{ts:.3f}.jpg")
            cv2.imwrite(path, frame)
            saved[ts] = path
    finally:
        cap.release()
    return saved