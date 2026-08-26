"""Layer 10: sponsor-language spotting over the ASR transcript. This is
the PRIMARY detector for long-form video (see asr.py docstring) -- it's
what makes a 90s sponsor read with a completely static visual detectable
at all, since a frame-only pipeline has zero signal for that case.

Two pattern sets on purpose: a LAUNCH phrase ("sponsored by") and a CTA
phrase ("use code", "% off") are different kinds of evidence and a
segment can have either, both, or neither -- keeping them separate lets
fusion (a later layer) weigh them differently rather than collapsing
everything into one "is this sponsor-y" bucket.

SPONSOR_LAUNCH_PATTERNS / SPONSOR_CTA_PATTERNS are a starting lexicon,
not a claim of completeness -- worth a line in DESIGN.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.asr import TranscriptSegment

SPONSOR_LAUNCH_PATTERNS = [
    r"\bsponsored by\b",
    r"\bbrought to you by\b",
    r"\bthanks? to (our|today'?s) sponsor\b",
    r"\bin partnership with\b",
    r"\btoday'?s (episode|video) is sponsored\b",
]

SPONSOR_CTA_PATTERNS = [
    r"\buse code\b",
    r"\bpromo code\b",
    r"\bdiscount code\b",
    r"%\s*off",
    r"\blink in the description\b",
    r"\blink in bio\b",
    r"\baffiliate link\b",
    r"\bcheck out the link\b",
]

_LAUNCH_RE = re.compile("|".join(SPONSOR_LAUNCH_PATTERNS), re.IGNORECASE)
_CTA_RE = re.compile("|".join(SPONSOR_CTA_PATTERNS), re.IGNORECASE)


@dataclass
class TranscriptHit:
    start_s: float
    end_s: float
    text: str
    is_launch: bool
    is_cta: bool


def find_sponsor_language(segments: list[TranscriptSegment]) -> list[TranscriptHit]:
    hits = []
    for seg in segments:
        is_launch = bool(_LAUNCH_RE.search(seg.text))
        is_cta = bool(_CTA_RE.search(seg.text))
        if is_launch or is_cta:
            hits.append(TranscriptHit(
                start_s=seg.start_s, end_s=seg.end_s, text=seg.text,
                is_launch=is_launch, is_cta=is_cta,
            ))
    return hits