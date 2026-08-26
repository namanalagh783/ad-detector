"""Layer 4: scene-cut detection using ffmpeg's built-in `scene` score.

This is deliberately cheap (no ML) -- it exists purely to tell later
layers (sampling.py) WHERE to look, not to detect ads itself.

EMPIRICAL NOTE: ffmpeg's commonly-cited example threshold of 0.3 is NOT a
safe default. Scene score scales with color/luminance delta, not with
"is this a genuine cut" -- a cut between two similarly-dark colors can
score well under 0.3 while a cut into something bright scores far above
it, even though both are equally real cuts. 0.08 was the threshold that
reliably caught known cuts in testing; it's a real precision/recall
trade-off (lower = catches subtler cuts, but fires more on noisy
footage), not a solved problem -- worth a line in DESIGN.md.
"""

from __future__ import annotations

import re
import subprocess

_PTS_RE = re.compile(r"pts_time:([0-9.]+)")


def detect_scene_cuts(video_path: str, threshold: float = 0.08) -> list[float]:
    """Returns sorted, deduplicated timestamps (seconds) where ffmpeg's
    scene score exceeds `threshold`."""
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-f", "null", "-",
    ]
    # ffmpeg writes showinfo/select output to stderr, not stdout.
    result = subprocess.run(cmd, capture_output=True, text=True)
    timestamps = [float(m) for m in _PTS_RE.findall(result.stderr)]
    return sorted(set(round(t, 3) for t in timestamps))