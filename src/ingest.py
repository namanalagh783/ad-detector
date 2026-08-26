"""Layers 2-3: real ingestion for local media files AND real YouTube
download.

Handles: platform/kind detection from a URL or path, real metadata
(duration, fps, resolution) via ffprobe, and (layer 3) actually
downloading a YouTube video via yt-dlp.

IMPORTANT CAVEAT ON THE YOUTUBE PATH: it was written and reasoned through
carefully, but could NOT be tested end-to-end in the environment this was
built in (no network access to youtube.com there). The local-file path
(layer 2) is fully verified; the YouTube path (layer 3) needs to be
verified against a real URL by whoever runs this next. See
tests/test_ingest_youtube.py, which is a manual/network-required test,
deliberately kept separate from the main offline test suite.

Instagram is NOT downloaded here -- per the brief's acquisition rule,
Reels always arrive as an already-acquired local file (screen capture is
explicitly acceptable), so INSTAGRAM always goes through the same local-
file path as FILE.
"""

from __future__ import annotations

import json
import os
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
    it needs an explicit hint or yt-dlp's info dict."""
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


def download_youtube(url: str, out_dir: str) -> tuple[str, dict]:
    """Download a YouTube video via yt-dlp. Returns (local_path, info_dict).

    UNTESTED in the environment this was written in -- see module
    docstring. Capped at 720p to keep downloads fast and cheap; the
    pipeline doesn't need source resolution for ad detection.
    """
    import yt_dlp  # imported lazily so layer 2 (local-only) never needs this dependency at all

    os.makedirs(out_dir, exist_ok=True)
    ydl_opts = {
        "outtmpl": os.path.join(out_dir, "%(id)s.%(ext)s"),
        "format": "bv*[height<=720]+ba/b[height<=720]",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        local_path = ydl.prepare_filename(info)
        # merge_output_format can change the extension after download
        # (e.g. source was .webm, output got merged to .mp4)
        if not os.path.exists(local_path):
            candidate = os.path.splitext(local_path)[0] + ".mp4"
            if os.path.exists(candidate):
                local_path = candidate
    return local_path, info


def _kind_from_youtube_info(url: str, info: dict, kind_hint: Kind | None) -> Kind:
    if kind_hint is not None:
        return kind_hint
    shape_guess = detect_kind_from_url(url)
    if shape_guess is not None:
        return shape_guess
    # URL shape didn't tell us (plain watch?v= URL) -- ask yt-dlp's own
    # metadata instead, which DOES know live vs. vod.
    if info.get("is_live"):
        return Kind.LIVE
    return Kind.VOD


def ingest_source(
    source: str,
    platform_hint: Platform | None = None,
    kind_hint: Kind | None = None,
    download_dir: str = "data/videos",
) -> MediaAsset:
    """Main entry point. Routes to real YouTube download or real local
    ffprobe ingestion depending on detected/hinted platform."""
    platform = platform_hint or detect_platform(source)

    if platform == Platform.YOUTUBE:
        local_path, info = download_youtube(source, download_dir)
        meta = probe_media(local_path)
        kind = _kind_from_youtube_info(source, info, kind_hint)
        return MediaAsset(
            source_url=source,
            platform=platform,
            kind=kind,
            local_path=local_path,
            duration_s=meta["duration_s"],
            fps=meta["fps"],
            width=meta["width"],
            height=meta["height"],
        )

    # FILE, INSTAGRAM (manually-acquired local Reel), or OTHER all resolve
    # to "read a local path".
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