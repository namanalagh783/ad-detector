"""Central home for pipeline tunables. Every number here is a decision
that should be defensible in DESIGN.md, not a number buried inside logic
where it's easy to forget it was a choice at all.
"""

from dataclasses import dataclass, field


@dataclass
class SamplingConfig:
    # Backbone (uniform) sampling interval in seconds, per video kind.
    # Shorts get a much denser backbone than long-form because an entire
    # ad can start and end within a couple of seconds; long-form leans on
    # ASR (a later layer) to catch ads during long static stretches
    # instead of raising the visual sampling rate everywhere.
    backbone_interval_s: dict = field(
        default_factory=lambda: {"vod": 5.0, "short": 0.5, "live": 3.0}
    )

    # Extra samples taken at these offsets (seconds) after every detected
    # scene cut. This is the direct fix for a short visual event (e.g. a
    # 1.4s animated bumper) landing entirely between two backbone samples
    # and never getting looked at.
    cut_offsets_s: tuple = (0.0, 0.3, 0.8, 1.3)

    # Hard cap on total frames sampled per video, independent of length,
    # so a pathological video (cuts every second) can't silently blow
    # past a reasonable processing budget.
    max_frames_per_video: int = 400

@dataclass
class ASRConfig:
    model_size: str = "small"  # faster-whisper model size
    device: str = "cpu"
    compute_type: str = "int8"
    # If set, load weights from this local directory instead of pulling
    # from huggingface.co at runtime -- useful for reproducible offline
    # runs, or in an environment with restricted network egress (this
    # config option exists because ASR was written in a sandbox that
    # could not reach huggingface.co at all -- see src/asr.py docstring).
    model_dir: str = ""

@dataclass
class VLMConfig:
    # gemini-3.6-flash: free-tier eligible as of when this was written.
    # Google's free-tier model list shifts over time (Pro lost free-tier
    # access entirely in April 2026) -- if this stops being free, check
    # https://ai.google.dev/gemini-api/docs/pricing for the current list.
    model: str = "gemini-3.6-flash"
    # Hard cap on VLM calls per video -- this, combined with only ever
    # calling classify_candidates() on a pre-filtered subset of frames
    # (never the full sample plan), is the actual mechanism keeping
    # stats.estimated_cost_usd bounded. See DESIGN.md for what we'd cut
    # first to go 10x cheaper (answer: scene-cut-only candidates with no
    # corroborating ASR/OCR signal -- those are the least certain ones).
    max_calls_per_video: int = 60
    # $0 because this runs on Gemini's free tier -- accurate for now, but
    # worth being honest in DESIGN.md that a free tier has rate-limit and
    # availability trade-offs a paid tier wouldn't. Update this if you
    # switch to a paid model or exceed free-tier quota.
    est_cost_per_call_usd: float = 0.0

@dataclass
class Config:
    """Bundles the three per-stage configs into one object pipeline.py
    can pass around, so a single caller (or a future CLI flag) can swap
    all settings for a video at once rather than threading three separate
    config objects through every function signature."""
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)


CONFIG = Config()