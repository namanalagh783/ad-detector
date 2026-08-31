# Ad Segment Detection Pipeline

Detects and timestamps advertising segments (sponsor reads, bumpers,
product placement) across long-form YouTube video, YouTube Shorts, and
Instagram Reels, by fusing four independent signals: speech transcript
analysis, on-screen text (OCR), a vision-language model's judgment on
candidate frames, and scene-cut-driven adaptive sampling.

Built incrementally, one tested layer at a time — see `DESIGN.md` for
architecture and reasoning, `EVAL.md` for metrics and results.

## Status

**Code: complete** (all required Tier 1 components built and tested).

**Docs: in progress.** `DESIGN.md` and `EVAL.md` exist with all objective
content filled in (architecture, real bugs found and fixed, real cost/
latency data); the ad taxonomy, the 8 required ambiguity-pack rulings,
`ground_truth.json`, the per-video results table, and the 3 failure-case
writeups are still TODO — clearly marked as such in those files.

**Known issue, not yet resolved:** VLM calls via Gemini's free tier are
slow and occasionally return `503 (high demand)` errors — observed
latencies from 5s to 22s per call. In one full real run, this was 91% of
total wall-clock time. See `DESIGN.md` Section 5 for details; a fix
(paid tier, or retry/backoff) is planned but not yet implemented.

**Not yet built:** live-stream windowing (Tier 2, optional).

## Prerequisites

- Python 3.11+
- [ffmpeg](https://www.gyan.dev/ffmpeg/builds/) (must be on PATH)
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) (must be
  on PATH, or set `pytesseract.pytesseract.tesseract_cmd` explicitly at
  the top of `src/ocr.py` if PATH detection doesn't work)
- A free [Gemini API key](https://aistudio.google.com/app/apikey) (no
  card required)
- Node.js — optional, only needed to run the viewer's logic test

## Setup

```bash
git clone <this repo>
cd ad-detection

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your key:
```
GEMINI_API_KEY=your_actual_key_here
```

## Project structure

```
src/
  schema.py     output contract (pydantic models)
  config.py     all tunable settings, with reasoning in comments
  ingest.py     local file + real YouTube ingestion
  scenes.py     scene-cut detection
  sampling.py   adaptive frame sampling
  frames.py     frame extraction
  ocr.py        on-screen text + ad-indicator matching
  audio.py      audio track extraction
  asr.py        speech-to-text (faster-whisper)
  signals.py    sponsor-language spotting over transcripts
  vlm.py        vision-model frame classification (Gemini)
  fusion.py     combines all signals into final segments
  pipeline.py   orchestrates the above end-to-end
  api.py        FastAPI HTTP endpoint
  env.py        .env loading helper
eval/
  metrics.py    temporal IoU, boundary error, precision/recall
  run_eval.py   CLI: compares predictions against ground_truth.json
tests/          one test file per component, plus fixtures/
viewer/         minimal HTML seek-bar viewer (no server needed)
scripts/        run_once.py (run the pipeline, save JSON output)
```

## Generating test fixtures

Most tests need small synthetic video fixtures. Generate them all in one
pass:

```bash
python tests/fixtures/generate_all.py
```

One fixture needs real speech and isn't reliably cross-platform from
Python alone — generate it separately (Windows, built-in TTS, no extra
install needed):
```powershell
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SetOutputToWaveFile("tests\fixtures\speech_fixture.wav")
$synth.Speak("This video is sponsored by Acme VPN. Use promo code SAVE TEN at checkout for twenty percent off.")
$synth.Dispose()
```

## Running tests

**Fully offline, no network or API key needed:**
```bash
python tests/test_schema.py
python tests/test_ingest.py
python tests/test_scenes.py
python tests/test_sampling.py
python tests/test_frames.py
python tests/test_ocr.py
python tests/test_audio.py
python tests/test_signals.py
python tests/test_fusion.py
python tests/test_pipeline_full.py
python tests/test_api.py
python tests/test_eval_metrics.py
python tests/test_eval_run.py
node tests/test_viewer_logic.mjs   # needs Node, optional
```

**Manual tests, need real network / a real API key / real downloaded
content:**
```bash
python tests/test_ingest_youtube.py   # real network, downloads a real video
python tests/test_asr.py              # real network on first run (model download)
python tests/test_vlm.py              # needs GEMINI_API_KEY
```

## Running the pipeline

**One-off, from the command line:**
```bash
python scripts/run_once.py <video_path_or_youtube_url> result.json
```

**As an HTTP API:**
```bash
uvicorn src.api:app --reload
```
Then `POST /analyze` with `{"url": "...", "platform": "...", "kind": "..."}`,
or open `http://127.0.0.1:8000/docs` for an interactive explorer.

## Running the evaluation

Once `ground_truth.json` exists and predictions have been generated for
each test-set video (via `scripts/run_once.py`, saved as
`data/outputs/<video_id>.json`):
```bash
python -m eval.run_eval --gt ground_truth.json --pred data/outputs/
```

## Using the viewer

No server needed. Open `viewer/index.html` directly in a browser, then:
1. "Load video" -> pick any local video file
2. "Load result JSON" -> pick a JSON file produced by `scripts/run_once.py`
   or the API

Detected segments render as colored blocks on the seek bar and as a
table below; clicking either jumps the video to that timestamp.