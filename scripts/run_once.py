"""Run the full real pipeline against a video and save the JSON result
to disk, for loading into the viewer (or just eyeballing).

Usage: python scripts/run_once.py <path_or_url> [output.json]
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env import load_env
load_env()

from src.asr import transcribe
from src.audio import extract_audio
from src.config import CONFIG
from src.frames import extract_frames
from src.fusion import fuse
from src.ingest import ingest_source
from src.ocr import ocr_frames
from src.pipeline import _select_vlm_candidates
from src.sampling import build_sample_plan
from src.scenes import detect_scene_cuts
from src.schema import AdDetectionResult, Source, Stats
from src.vlm import classify_candidates
from datetime import datetime, timezone


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_once.py <video_path_or_url> [output.json]")
        sys.exit(1)

    source = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "result.json"
    t_start = time.time()

    def step(label):
        print(f"[{time.time() - t_start:5.1f}s] {label}")

    step(f"Ingesting: {source}")
    asset = ingest_source(source, download_dir="data/videos")

    step("Detecting scene cuts")
    scene_cuts = detect_scene_cuts(asset.local_path)
    print(f"         -> {len(scene_cuts)} cuts found")

    step("Building sample plan")
    sample_plan = build_sample_plan(asset.duration_s, asset.kind, scene_cuts, CONFIG.sampling)
    print(f"         -> {len(sample_plan)} frames planned")

    step("Extracting frames")
    frame_paths = extract_frames(asset.local_path, sample_plan, "data/frames")

    step("Running OCR")
    ocr_hits = ocr_frames(frame_paths)

    step("Extracting audio")
    audio_path = extract_audio(asset.local_path, "data/audio")

    step("Transcribing (this is usually the slow one -- CPU Whisper)")
    transcript_segments = transcribe(audio_path, CONFIG.asr)
    print(f"         -> {len(transcript_segments)} transcript segments")

    step("Selecting VLM candidates + classifying (real API calls)")
    candidates = _select_vlm_candidates(frame_paths, transcript_segments, ocr_hits, scene_cuts)
    print(f"         -> {len(candidates)} candidate frames to classify")
    vlm_results = classify_candidates(candidates, CONFIG.vlm)

    step("Fusing signals")
    segments = fuse(transcript_segments, ocr_hits, vlm_results, scene_cuts, sample_plan)

    model_calls = len([r for r in vlm_results if r.error is None])
    result = AdDetectionResult(
        source=Source(
            url=asset.source_url, platform=asset.platform, kind=asset.kind,
            duration_s=asset.duration_s, processed_at=datetime.now(timezone.utc),
        ),
        segments=segments,
        stats=Stats(
            wall_clock_s=round(time.time() - t_start, 2),
            estimated_cost_usd=round(model_calls * CONFIG.vlm.est_cost_per_call_usd, 4),
            frames_sampled=len(frame_paths),
            model_calls=model_calls,
        ),
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result.model_dump_json(indent=2))

    step(f"Done. Saved to {out_path}")
    print(f"\nFound {len(result.segments)} segment(s):")
    for seg in result.segments:
        print(f"  {seg.start_s:.1f}-{seg.end_s:.1f}s  {seg.ad_type.value}  brand={seg.brand}  confidence={seg.confidence}")


if __name__ == "__main__":
    main()