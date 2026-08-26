"""Layer 11: VLM classification on CANDIDATE frames only -- never every
sampled frame. A frame becomes a candidate via a cheaper signal first
(ASR transcript hit, OCR ad-indicator text) or, as a fallback, proximity
to a scene cut with no other signal -- see fusion.py (a later layer) for
how candidates actually get selected. This module just does the
classification call itself and enforces the cost cap.

Uses Gemini (free tier) rather than a paid-only provider -- see
src/config.py's VLMConfig for the model choice and its trade-offs.

CAVEAT: not run end-to-end with a real API key in the environment this
was written in (no key was available there). Needs a real key to verify
for real.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from src.config import VLMConfig

_SYSTEM_PROMPT = """You are analyzing a single video frame to help detect advertising content in a video pipeline.
Respond with ONLY a JSON object and nothing else -- no markdown fences, no preamble. Fields:
- "is_ad_visual": boolean -- true if this frame shows clear advertising/sponsor visual content (product shot, sponsor bumper/card, discount code overlay, prominent brand logo, "sponsored" label, etc). False for ordinary talking-head or unrelated content.
- "brand": string or null -- a brand/product name if visible, else null.
- "description": string -- one short, plain sentence describing what's on screen.
- "confidence": number 0.0-1.0 -- your confidence in is_ad_visual.
"""


@dataclass
class VlmFrameResult:
    timestamp_s: float
    is_ad_visual: bool
    brand: Optional[str]
    description: str
    confidence: float
    error: Optional[str] = None


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text.strip()


def classify_frame(client, image_path: str, timestamp_s: float, model: str) -> VlmFrameResult:
    from google.genai import types  # lazy import: earlier layers never need this

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            "Analyze this frame.",
        ],
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            response_mime_type="application/json",  # asks Gemini to guarantee valid JSON syntax
        ),
    )
    parsed = json.loads(_strip_fences(response.text))
    return VlmFrameResult(
        timestamp_s=timestamp_s,
        is_ad_visual=bool(parsed.get("is_ad_visual", False)),
        brand=parsed.get("brand"),
        description=parsed.get("description", ""),
        confidence=float(parsed.get("confidence", 0.5)),
    )


def classify_candidates(
    image_paths: dict[float, str],
    cfg: VLMConfig,
    client=None,
) -> list[VlmFrameResult]:
    """image_paths should already be pre-filtered to candidates by the
    caller -- this function just enforces the hard cost cap and never
    raises on a single-frame failure (a bad frame shouldn't kill the
    whole video's pipeline run)."""
    if client is None:
        import os

        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "No GEMINI_API_KEY or GOOGLE_API_KEY found in the environment. "
                "Set one in a .env file (see src/env.py) or export it directly."
            )
        # Passed explicitly rather than relying on genai.Client()'s
        # automatic env-var detection -- that auto-detection did not
        # reliably pick up the key in testing, despite the variable being
        # confirmed present in os.environ at call time.
        client = genai.Client(api_key=api_key)

    results = []
    for ts, path in list(image_paths.items())[: cfg.max_calls_per_video]:
        try:
            results.append(classify_frame(client, path, ts, cfg.model))
        except Exception as e:  # noqa: BLE001 -- deliberately broad, see docstring
            results.append(VlmFrameResult(
                timestamp_s=ts, is_ad_visual=False, brand=None,
                description="", confidence=0.0, error=str(e),
            ))
    return results