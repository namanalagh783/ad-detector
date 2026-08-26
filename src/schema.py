"""Layer 0: the output data contract for ad-segment detection.

Everything downstream (ingestion, detection, API) produces or consumes an
AdDetectionResult. Nothing in this module knows how segments are found.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class Platform(str, Enum):
    YOUTUBE = "youtube"
    INSTAGRAM = "instagram"
    FILE = "file"
    OTHER = "other"


class Kind(str, Enum):
    VOD = "vod"
    SHORT = "short"
    LIVE = "live"


class AdType(str, Enum):
    PREROLL = "preroll"
    MIDROLL_SPONSOR_READ = "midroll_sponsor_read"
    PRODUCT_PLACEMENT = "product_placement"
    SELF_PROMO = "self_promo"
    AFFILIATE = "affiliate"
    PLATFORM_INSERTED = "platform_inserted"
    BUMPER = "bumper"
    OTHER = "other"


class Signal(str, Enum):
    ASR = "asr"
    OCR = "ocr"
    VLM_FRAME = "vlm_frame"
    SCENE_CUT = "scene_cut"


class Source(BaseModel):
    url: str
    platform: Platform
    kind: Kind
    duration_s: float
    processed_at: datetime


class Evidence(BaseModel):
    frame_timestamps: list[float]
    transcript_span: Optional[str] = None
    signals_used: list[Signal]


class Segment(BaseModel):
    id: str
    start_s: float
    end_s: float
    ad_type: AdType
    confidence: float = Field(ge=0.0, le=1.0)
    brand: Optional[str] = None
    description: str
    evidence: Evidence

    @model_validator(mode="after")
    def _check_span(self) -> "Segment":
        if self.end_s < self.start_s:
            raise ValueError(
                f"end_s ({self.end_s}) must be >= start_s ({self.start_s})"
            )
        return self


class Stats(BaseModel):
    wall_clock_s: float
    estimated_cost_usd: float
    frames_sampled: int
    model_calls: int


class AdDetectionResult(BaseModel):
    source: Source
    segments: list[Segment]
    stats: Stats
