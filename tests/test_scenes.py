"""Layer 4 checks: scene-cut detection against a fixture with known cuts
at 4.0s (blue->yellow) and 6.0s (yellow->green).

Runnable directly (`python tests/test_scenes.py`) or under pytest.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.scenes import detect_scene_cuts

FIXTURE = str(Path(__file__).parent / "fixtures" / "scenes_fixture.mp4")


def test_detects_both_known_cuts():
    cuts = detect_scene_cuts(FIXTURE)
    assert len(cuts) >= 2, f"expected at least 2 cuts, got {cuts}"
    assert any(abs(c - 4.0) < 0.2 for c in cuts), f"missed cut near 4.0s, got {cuts}"
    assert any(abs(c - 6.0) < 0.2 for c in cuts), f"missed cut near 6.0s, got {cuts}"


def test_no_false_cuts_on_flat_color():
    # a static single-color clip should produce zero cuts
    flat = str(Path(__file__).parent / "fixtures" / "sample.mp4")
    cuts = detect_scene_cuts(flat)
    assert cuts == [], f"expected no cuts on flat footage, got {cuts}"


def test_higher_threshold_is_stricter():
    loose = detect_scene_cuts(FIXTURE, threshold=0.05)
    strict = detect_scene_cuts(FIXTURE, threshold=0.9)
    assert len(strict) <= len(loose), "a stricter threshold should never find MORE cuts"


CHECKS = [
    ("detects both known cuts (4.0s, 6.0s)", test_detects_both_known_cuts),
    ("no false cuts on flat/static footage", test_no_false_cuts_on_flat_color),
    ("higher threshold is stricter (finds <= cuts)", test_higher_threshold_is_stricter),
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