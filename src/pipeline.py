"""Layer 13: the full pipeline wired for real, replacing every remaining
stub from layer 1's walking skeleton. Calls every layer built so far, in
order: ingest -> scene cuts -> sampling -> frames -> OCR -> audio
extraction -> ASR -> VLM (on selected candidates only) -> fusion ->
output contract.

CAVEAT: ASR needs huggingface.co and VLM needs a real API key, neither
available in the sandbox this was built in -- so this file's own test
mocks those two specific calls (transcribe, classify_candidates) while
using REAL scene detection, REAL sampling, REAL frame extraction, and
REAL OCR against an actual fixture video. Both ASR and VLM were already
independently confirmed working for real elsewhere (layers 9 and 11) --
this layer's job is proving the WIRING is correct, not re-proving those
two components work.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

from src.asr import TranscriptSegment, transcribe
from src.audio import extract_audio
from src.config import CONFIG, Config
from src.frames import extract_frames
from src.fusion import fuse
from src.ingest import MediaAsset, ingest_source
from src.ocr import OcrHit, ocr_frames
from src.sampling import build_sample_plan
from src.scenes import detect_scene_cuts
from src.schema import AdDetectionResult, Kind, Platform, Source, Stats
from src.signals import find_sponsor_language
from src.vlm import classify_candidates


def _select_vlm_candidates(
    frame_paths: dict[float, str],
    transcript_segments: list[TranscriptSegment],
    ocr_hits: list[OcrHit],
    scene_cuts: list[float],
) -> dict[float, str]:
    """Which sampled frames actually get a (costed) VLM call. Priority:
    frames near a transcript sponsor-language hit, or frames OCR already
    flagged as ad-indicator text -- both are cheap signals with
    corroborating evidence already. Fallback, only if NEITHER cheaper
    signal found anything at all: frames near a scene cut, so a silent,
    textless ad visual isn't invisible to every signal at once."""
    hit_windows = [
        (h.start_s - 1.0, h.end_s + 1.0) for h in find_sponsor_language(transcript_segments)
    ]
    ocr_flagged = {h.timestamp_s for h in ocr_hits if h.is_ad_indicator}

    def in_any_window(ts: float) -> bool:
        return any(lo <= ts <= hi for lo, hi in hit_windows)

    candidates = {
        ts: p for ts, p in frame_paths.items()
        if ts in ocr_flagged or in_any_window(ts)
    }
    if not candidates:
        candidates = {
            ts: p for ts, p in frame_paths.items()
            if any(abs(ts - c) <= 1.5 for c in scene_cuts)
        }
    return candidates


def run_pipeline(
    source: str,
    out_dir: str = "data",
    platform_hint: Optional[Platform] = None,
    kind_hint: Optional[Kind] = None,
    cfg: Config = CONFIG,
    skip_vlm: bool = False,
) -> AdDetectionResult:
    t0 = time.time()

    asset: MediaAsset = ingest_source(
        source, platform_hint=platform_hint, kind_hint=kind_hint,
        download_dir=f"{out_dir}/videos",
    )

    scene_cuts = detect_scene_cuts(asset.local_path)
    sample_plan = build_sample_plan(asset.duration_s, asset.kind, scene_cuts, cfg.sampling)
    frame_paths = extract_frames(asset.local_path, sample_plan, f"{out_dir}/frames")

    ocr_hits = ocr_frames(frame_paths)

    audio_path = extract_audio(asset.local_path, f"{out_dir}/audio")
    transcript_segments = transcribe(audio_path, cfg.asr)

    vlm_results = []
    model_calls = 0
    if not skip_vlm:
        candidate_frames = _select_vlm_candidates(frame_paths, transcript_segments, ocr_hits, scene_cuts)
        vlm_results = classify_candidates(candidate_frames, cfg.vlm)
        model_calls = len([r for r in vlm_results if r.error is None])

    segments = fuse(transcript_segments, ocr_hits, vlm_results, scene_cuts, sample_plan)

    wall_clock_s = time.time() - t0
    estimated_cost_usd = round(model_calls * cfg.vlm.est_cost_per_call_usd, 4)

    return AdDetectionResult(
        source=Source(
            url=asset.source_url,
            platform=asset.platform,
            kind=asset.kind,
            duration_s=asset.duration_s,
            processed_at=datetime.now(timezone.utc),
        ),
        segments=segments,
        stats=Stats(
            wall_clock_s=round(wall_clock_s, 2),
            estimated_cost_usd=estimated_cost_usd,
            frames_sampled=len(frame_paths),
            model_calls=model_calls,
        ),
    )