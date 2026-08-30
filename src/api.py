"""Layer 14: the HTTP API. One endpoint, per the assignment's Tier 1
requirement: "Ship a working HTTP API with at least one endpoint that
takes a URL and returns the result."

Run with: uvicorn src.api:app --reload
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.pipeline import run_pipeline
from src.schema import AdDetectionResult, Kind, Platform

app = FastAPI(title="Ad Segment Detector", version="0.1.0")


class AnalyzeRequest(BaseModel):
    url: str
    platform: Optional[Platform] = None
    kind: Optional[Kind] = None
    skip_vlm: bool = False


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AdDetectionResult)
def analyze(req: AnalyzeRequest) -> AdDetectionResult:
    try:
        return run_pipeline(
            source=req.url,
            platform_hint=req.platform,
            kind_hint=req.kind,
            skip_vlm=req.skip_vlm,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001 -- surfaces pipeline errors as a clean 500 instead of a stack trace leak
        raise HTTPException(status_code=500, detail=str(e)) from e