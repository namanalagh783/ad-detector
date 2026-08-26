"""Layer 1: the walking skeleton.

A single fully-fake end-to-end pipeline. No ffmpeg, no whisper, no
yt-dlp, no real detection -- every stage is stubbed. The point of this
layer is to prove the orchestration shape (ingest -> detect -> stats ->
AdDetectionResult) works before any stage has real complexity in it.
Later layers replace one stub at a time with a real implementation;
this function's shape shouldn't need to change as that happens.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from src.schema import (
    AdDetectionResult,
    AdType,
    Evidence,
    Kind,
    Platform,
    Segment,
    Signal,
    Source,
    Stats,
)


def _fake_ingest(source: str, platform_hint: Platform | None, kind_hint: Kind | None) -> Source:
    """Stub: no filesystem/network access yet. Just echoes back what
    we were given, with sane defaults filled in."""
    return Source(
        url=source,
        platform=platform_hint or Platform.OTHER,
        kind=kind_hint or Kind.VOD,
        duration_s=100.0,  # placeholder, replaced by real ffprobe output later
        processed_at=datetime.now(timezone.utc),
    )


def _fake_detect(source: Source) -> list[Segment]:
    """Stub: no real signal processing yet. Returns one hardcoded segment
    so the full Segment/Evidence object graph gets exercised, not just
    an empty list."""
    return [
        Segment(
            id="seg_01",
            start_s=24.0,
            end_s=87.5,
            ad_type=AdType.MIDROLL_SPONSOR_READ,
            confidence=0.81,
            brand="Acme VPN",
            description="[stub] placeholder segment from the walking skeleton, not a real detection.",
            evidence=Evidence(
                frame_timestamps=[25.0, 40.5, 71.0],
                transcript_span="[stub transcript span]",
                signals_used=[Signal.ASR, Signal.OCR, Signal.VLM_FRAME, Signal.SCENE_CUT],
            ),
        )
    ]


def run_pipeline(
    source: str,
    platform_hint: Platform | None = None,
    kind_hint: Kind | None = None,
) -> AdDetectionResult:
    t0 = time.time()

    src = _fake_ingest(source, platform_hint, kind_hint)
    segments = _fake_detect(src)

    wall_clock_s = time.time() - t0

    return AdDetectionResult(
        source=src,
        segments=segments,
        stats=Stats(
            wall_clock_s=round(wall_clock_s, 4),
            estimated_cost_usd=0.0,
            frames_sampled=0,
            model_calls=0,
        ),
    )