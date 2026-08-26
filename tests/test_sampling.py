"""Layer 5 checks: adaptive sampling plan against known durations/cuts.

Runnable directly (`python tests/test_sampling.py`) or under pytest.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import SamplingConfig
from src.sampling import build_sample_plan
from src.schema import Kind


def test_backbone_covers_full_duration_no_cuts():
    plan = build_sample_plan(20.0, Kind.VOD, [], SamplingConfig())
    assert 0.0 in plan
    assert plan[-1] < 20.0
    assert len(plan) == 4


def test_short_has_denser_backbone_than_vod():
    vod_plan = build_sample_plan(20.0, Kind.VOD, [], SamplingConfig())
    short_plan = build_sample_plan(20.0, Kind.SHORT, [], SamplingConfig())
    assert len(short_plan) > len(vod_plan)


def test_samples_cluster_around_scene_cut():
    cfg = SamplingConfig()
    plan = build_sample_plan(20.0, Kind.VOD, [10.0], cfg)
    for offset in cfg.cut_offsets_s:
        expected = round(10.0 + offset, 3)
        assert expected in plan, f"expected sample at {expected}, plan={plan}"


def test_cut_near_end_of_video_doesnt_overshoot():
    cfg = SamplingConfig()
    plan = build_sample_plan(10.5, Kind.VOD, [10.0], cfg)
    assert all(t <= 10.5 for t in plan), f"plan overshoots duration: {plan}"


def test_budget_cap_is_respected():
    cfg = SamplingConfig(max_frames_per_video=10)
    many_cuts = [float(i) for i in range(50)]
    plan = build_sample_plan(50.0, Kind.SHORT, many_cuts, cfg)
    assert len(plan) <= 10, f"plan exceeded budget: {len(plan)} frames"


def test_thinning_protects_cut_triggered_samples():
    cfg = SamplingConfig(max_frames_per_video=8)
    cuts = [5.0]
    plan = build_sample_plan(20.0, Kind.VOD, cuts, cfg)
    for offset in cfg.cut_offsets_s:
        expected = round(5.0 + offset, 3)
        assert expected in plan, f"cut-triggered sample {expected} was thinned away, plan={plan}"


CHECKS = [
    ("backbone covers full duration with no cuts", test_backbone_covers_full_duration_no_cuts),
    ("SHORT backbone denser than VOD", test_short_has_denser_backbone_than_vod),
    ("samples cluster around a scene cut", test_samples_cluster_around_scene_cut),
    ("cut near end of video doesn't overshoot duration", test_cut_near_end_of_video_doesnt_overshoot),
    ("budget cap is respected under heavy cuts", test_budget_cap_is_respected),
    ("thinning protects cut-triggered samples over backbone", test_thinning_protects_cut_triggered_samples),
]


def main() -> int:
    failures = 0
    for name, check in CHECKS:
        try:
            check()
        except Exception as exc:  # noqa: BLE001 - report, don't stop
            failures += 1
            print(f"FAIL  {name}\n        {type(exc).__name__}: {exc}")
        else:
            print(f"PASS  {name}")
    print(f"\n{len(CHECKS) - failures}/{len(CHECKS)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())