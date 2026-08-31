"""Diagnostic: time each individual VLM call to see whether latency is
uniform (network/model latency) or bursty with long gaps (rate-limit
backoff). Reuses frames already extracted in the last run.

Usage: python scripts/diagnose_vlm_latency.py
"""

import glob
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env import load_env
load_env()

from src.config import CONFIG
from src.vlm import classify_frame
import os
from google import genai

api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

# grab whatever frames are sitting in data/frames from the last run
frame_files = sorted(glob.glob("data/frames/frame_*.jpg"))[:5]
print(f"Timing {len(frame_files)} individual calls...\n")

for path in frame_files:
    ts = float(Path(path).stem.replace("frame_", ""))
    t0 = time.time()
    try:
        result = classify_frame(client, path, ts, CONFIG.vlm.model)
        elapsed = time.time() - t0
        print(f"  {ts:6.2f}s  {elapsed:6.2f}s elapsed  is_ad_visual={result.is_ad_visual}")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  {ts:6.2f}s  {elapsed:6.2f}s elapsed  ERROR: {e}")