"""Layer 2: real ingestion for local media files.

Handles: platform/kind detection from a URL or path, and real metadata
(duration, fps, resolution) via ffprobe for local files. This covers
platform=file AND manually-acquired Instagram Reels (per the brief's
acquisition rule: Reels come in as a local file, never a live download).

YouTube download is deliberately NOT implemented here yet -- ingest_source
raises NotImplementedError for youtube.com URLs. That's layer 3 (needs
yt-dlp + real network access to youtube.com to verify it actually works).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from urllib.parse import urlparse

from src.schema import Kind, Platform


@dataclass
class MediaAsset:
    """Internal ingestion result -- not part of the output contract.
    pipeline.py will turn this into a schema.Source once ingestion is
    wired in (layer 3)."""
    source_url: str
    platform: Platform
    kind: Kind
    local_path: str
    duration_s: float
    fps: float
    width: int
    height: int


def detect_platform(source: str) -> Platform:
    """URL shape -> Platform. A bare local path (no scheme) is FILE."""
    if source.startswith(("http://", "https://")):
        host = urlparse(source).netloc.lower()
        if "youtube.com" in host or "youtu.be" in host:
            return Platform.YOUTUBE
        if "instagram.com" in host:
            return Platform.INSTAGRAM
        return Platform.OTHER
    return Platform.FILE


def detect_kind_from_url(url: str) -> Kind | None:
    """Best-effort kind guess from URL shape alone. Returns None when the
    shape doesn't tell us -- e.g. a plain youtube.com/watch?v=... URL
    could be a vod or a live broadcast; that's not decidable from the URL,
    it needs an explicit hint or (later, layer 3) yt-dlp's info dict."""
    if "/shorts/" in url:
        return Kind.SHORT
    if "instagram.com/reel" in url:
        return Kind.SHORT
    return None


def probe_media(path: str) -> dict:
    """ffprobe wrapper: duration_s, fps, width, height for a local file."""
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)

    duration_s = float(info["format"].get("duration", 0.0))
    width = height = 0
    fps = 0.0
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video":
            width = int(stream.get("width", 0))
            height = int(stream.get("height", 0))
            num, _, den = stream.get("avg_frame_rate", "0/1").partition("/")
            try:
                fps = float(num) / float(den) if float(den) else 0.0
            except (ValueError, ZeroDivisionError):
                fps = 0.0
            break

    return {"duration_s": duration_s, "fps": fps, "width": width, "height": height}


def ingest_source(
    source: str,
    platform_hint: Platform | None = None,
    kind_hint: Kind | None = None,
) -> MediaAsset:
    """Main entry point for this layer. Local files only -- see module
    docstring for why YouTube isn't handled here yet."""
    platform = platform_hint or detect_platform(source)

    if platform == Platform.YOUTUBE:
        raise NotImplementedError(
            "YouTube download isn't built yet (layer 3, needs yt-dlp). "
            "This layer only ingests local files."
        )

    # FILE, INSTAGRAM (manually-acquired local Reel), or OTHER all resolve
    # to "read a local path" at this layer.
    import os
    if not os.path.exists(source):
        raise FileNotFoundError(f"Local media not found: {source}")

    meta = probe_media(source)
    kind = kind_hint or detect_kind_from_url(source) or Kind.VOD

    return MediaAsset(
        source_url=source,
        platform=platform,
        kind=kind,
        local_path=source,
        duration_s=meta["duration_s"],
        fps=meta["fps"],
        width=meta["width"],
        height=meta["height"],
    )