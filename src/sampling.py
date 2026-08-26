"""Layer 5: adaptive frame sampling.

Decides WHICH timestamps in a video actually get looked at, combining:
1. A uniform "backbone" -- so long static stretches aren't left completely
   unobserved.
2. Densified sampling around every scene cut -- so short visual events
   aren't averaged away by a sparse uniform rate.

Explicitly NOT "uniform 1fps." See src/config.py for the tunables and
their reasoning.
"""

from __future__ import annotations

from src.config import SamplingConfig
from src.schema import Kind


def build_sample_plan(
    duration_s: float,
    kind: Kind,
    scene_cuts: list[float],
    cfg: SamplingConfig,
) -> list[float]:
    """Returns a sorted, deduplicated list of timestamps (seconds) to
    extract frames at."""
    backbone_interval = cfg.backbone_interval_s.get(kind.value, 5.0)

    timestamps: set[float] = {0.0}
    t = 0.0
    while t < duration_s:
        timestamps.add(round(t, 3))
        t += backbone_interval

    for cut in scene_cuts:
        for offset in cfg.cut_offsets_s:
            ts = cut + offset
            if 0.0 <= ts <= duration_s:
                timestamps.add(round(ts, 3))

    plan = sorted(timestamps)

    if len(plan) > cfg.max_frames_per_video:
        plan = _thin_backbone_first(plan, scene_cuts, cfg)

    return plan


def _thin_backbone_first(
    plan: list[float],
    scene_cuts: list[float],
    cfg: SamplingConfig,
) -> list[float]:
    """When over budget, drop backbone samples before ever dropping
    cut-triggered samples -- cut-triggered samples are doing the
    precision work (catching short events); backbone samples are just a
    safety net for otherwise-static stretches, so they're the cheaper
    thing to thin out."""
    cut_triggered = set()
    for cut in scene_cuts:
        for offset in cfg.cut_offsets_s:
            cut_triggered.add(round(cut + offset, 3))

    protected = sorted(t for t in plan if t in cut_triggered or t == 0.0)
    backbone_only = sorted(t for t in plan if t not in cut_triggered and t != 0.0)

    budget_left = max(cfg.max_frames_per_video - len(protected), 0)
    if budget_left <= 0 or not backbone_only:
        return protected[: cfg.max_frames_per_video]

    stride = max(1, len(backbone_only) // budget_left)
    thinned_backbone = backbone_only[::stride][:budget_left]

    return sorted(set(protected) | set(thinned_backbone))