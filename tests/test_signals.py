"""Layer 10 checks: sponsor-language spotting, including against the
EXACT real sentence your ASR just produced in layer 9 -- not hand-typed
fake data this time.

Runnable directly (`python tests/test_signals.py`) or under pytest.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.asr import TranscriptSegment
from src.signals import find_sponsor_language


def test_real_asr_sentence_flagged_as_launch():
    segments = [TranscriptSegment(0.0, 4.0, "this video is sponsored by acme vpn.")]
    hits = find_sponsor_language(segments)
    assert len(hits) == 1
    assert hits[0].is_launch is True
    assert hits[0].is_cta is False


def test_real_asr_sentence_flagged_as_cta():
    segments = [TranscriptSegment(4.0, 9.0, "use promo code save10 at checkout for 20% off.")]
    hits = find_sponsor_language(segments)
    assert len(hits) == 1
    assert hits[0].is_cta is True
    assert hits[0].is_launch is False


def test_ordinary_speech_not_flagged():
    segments = [
        TranscriptSegment(0.0, 3.0, "welcome back to the show today."),
        TranscriptSegment(3.0, 6.0, "let's get into the topic."),
    ]
    hits = find_sponsor_language(segments)
    assert hits == []


def test_segment_can_be_both_launch_and_cta():
    segments = [TranscriptSegment(
        0.0, 5.0, "brought to you by acme vpn, use code save10 for 20% off"
    )]
    hits = find_sponsor_language(segments)
    assert len(hits) == 1
    assert hits[0].is_launch is True
    assert hits[0].is_cta is True


def test_only_flagged_segments_returned():
    segments = [
        TranscriptSegment(0.0, 3.0, "just talking normally here."),
        TranscriptSegment(3.0, 6.0, "sponsored by acme vpn."),
        TranscriptSegment(6.0, 9.0, "back to the regular content."),
    ]
    hits = find_sponsor_language(segments)
    assert len(hits) == 1
    assert hits[0].start_s == 3.0


CHECKS = [
    ("real ASR sentence 'sponsored by acme vpn' flagged as launch", test_real_asr_sentence_flagged_as_launch),
    ("real ASR sentence 'use promo code...20% off' flagged as CTA", test_real_asr_sentence_flagged_as_cta),
    ("ordinary speech not flagged", test_ordinary_speech_not_flagged),
    ("a segment can be both launch and CTA at once", test_segment_can_be_both_launch_and_cta),
    ("only flagged segments are returned, not all segments", test_only_flagged_segments_returned),
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