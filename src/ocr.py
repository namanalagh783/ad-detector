"""Layer 7: OCR over sampled frames. Cheap and local (no API cost), so
this runs on every sampled frame -- it's what catches platform-native
"Sponsored"/"Paid partnership" labels, bumper card text, and discount
codes without needing a model call per frame.

AD_INDICATOR_PATTERNS is a starting lexicon, explicitly not a claim of
completeness -- worth a line in DESIGN.md about how it should grow
(e.g. platform-specific "Paid partnership with X" formats).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from PIL import Image

AD_INDICATOR_PATTERNS = [
    r"\bsponsor(ed)?\b",
    r"\bpaid partnership\b",
    r"\bad\b",
    r"\baffiliate\b",
    r"\bpromo(tion)?\b",
    r"\buse code\b",
    r"\bdiscount\b",
    r"%\s*off",
    r"\blink in bio\b",
    r"\bshop now\b",
]
_AD_INDICATOR_RE = re.compile("|".join(AD_INDICATOR_PATTERNS), re.IGNORECASE)


@dataclass
class OcrHit:
    timestamp_s: float
    text: str
    is_ad_indicator: bool


def ocr_frames(frame_paths: dict[float, str]) -> list[OcrHit]:
    hits = []
    for ts, path in frame_paths.items():
        text = pytesseract.image_to_string(Image.open(path)).strip()
        is_ad = bool(_AD_INDICATOR_RE.search(text)) if text else False
        hits.append(OcrHit(timestamp_s=ts, text=text, is_ad_indicator=is_ad))
    return hits