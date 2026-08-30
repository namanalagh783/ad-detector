"""Layer 14 checks: the /analyze endpoint via FastAPI's TestClient
(in-process, no real server or network needed). Same ASR/VLM mocking
approach as layer 13's test -- the API layer is thin, so this mostly
checks request/response wiring, status codes, and error handling, not
detection logic (already covered by test_pipeline_full.py).

Runnable directly (`python tests/test_api.py`) or under pytest.
"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from src.api import app
from src.asr import TranscriptSegment
from src.vlm import VlmFrameResult

FIXTURE = str(Path(__file__).parent / "fixtures" / "pipeline_fixture.mp4")

FAKE_TRANSCRIPT = [
    TranscriptSegment(0.0, 3.8, "welcome back to the show today."),
    TranscriptSegment(4.0, 7.0, "this video is sponsored by acme vpn."),
    TranscriptSegment(7.0, 9.0, "use promo code save10 for 20% off."),
    TranscriptSegment(9.0, 12.8, "anyway let's get back to it."),
]


def _fake_classify_candidates(image_paths, cfg, client=None):
    results = []
    for ts in image_paths:
        if 4.0 <= ts <= 9.0:
            results.append(VlmFrameResult(
                timestamp_s=ts, is_ad_visual=True, brand="Acme Vpn" if ts >= 5.4 else None,
                description="sponsor visual", confidence=0.85,
            ))
        else:
            results.append(VlmFrameResult(
                timestamp_s=ts, is_ad_visual=False, brand=None,
                description="ordinary content", confidence=0.7,
            ))
    return results


client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_analyze_returns_valid_schema_shaped_response():
    with patch("src.pipeline.transcribe", return_value=FAKE_TRANSCRIPT), \
         patch("src.pipeline.classify_candidates", side_effect=_fake_classify_candidates):
        resp = client.post("/analyze", json={"url": FIXTURE, "platform": "file", "kind": "vod"})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "source" in body and "segments" in body and "stats" in body
    assert len(body["segments"]) == 1
    assert body["segments"][0]["ad_type"] == "midroll_sponsor_read"
    assert body["segments"][0]["brand"] == "Acme Vpn"


def test_analyze_missing_local_file_returns_404():
    resp = client.post("/analyze", json={"url": "tests/fixtures/does_not_exist.mp4", "platform": "file"})
    assert resp.status_code == 404, resp.text


def test_analyze_malformed_request_returns_422():
    resp = client.post("/analyze", json={"platform": "file"})
    assert resp.status_code == 422, resp.text


def test_analyze_invalid_platform_value_returns_422():
    resp = client.post("/analyze", json={"url": FIXTURE, "platform": "not_a_real_platform"})
    assert resp.status_code == 422, resp.text


CHECKS = [
    ("GET /health returns ok", test_health_endpoint),
    ("POST /analyze returns a valid, schema-shaped response", test_analyze_returns_valid_schema_shaped_response),
    ("POST /analyze with missing local file returns 404", test_analyze_missing_local_file_returns_404),
    ("POST /analyze with missing required field returns 422", test_analyze_malformed_request_returns_422),
    ("POST /analyze with invalid enum value returns 422", test_analyze_invalid_platform_value_returns_422),
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