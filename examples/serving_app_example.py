"""Minimal FastAPI inference service with input validation, versioning and a fallback path.

Run locally:  uvicorn examples.serving_app_example:app --reload   (after wiring your pipeline/schema)
The factory `create_app` is what the tests exercise.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException

from dstk.contracts import validate_records


def create_app(pipeline, schema: Dict[str, Any], model_version: str,
               fallback_value: Optional[float] = None, max_batch: int = 1000) -> FastAPI:
    """pipeline: fitted estimator exposing .predict(DataFrame)
    schema: dstk.contracts schema describing ONE valid raw record
    fallback_value: returned (flagged) if the model raises; None means return HTTP 503 instead"""
    app = FastAPI(title="model service", version=model_version)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_version": model_version}

    @app.post("/predict")
    def predict(payload: dict) -> dict:
        records = payload.get("records")
        if not isinstance(records, list) or not records:
            raise HTTPException(status_code=422, detail=["'records' must be a non-empty list"])
        if len(records) > max_batch:
            raise HTTPException(status_code=413, detail=[f"batch larger than {max_batch}"])
        report = validate_records(records, schema)
        if not report.ok:
            raise HTTPException(status_code=422, detail=report.errors)
        started = time.perf_counter()
        try:
            preds = [float(p) for p in pipeline.predict(pd.DataFrame(records))]
            fallback = False
        except Exception as exc:  # noqa: BLE001 - the fallback path must catch model failures
            if fallback_value is None:
                raise HTTPException(status_code=503, detail=["model failure and no fallback configured"]) from exc
            preds, fallback = [float(fallback_value)] * len(records), True
        return {"predictions": preds, "model_version": model_version, "fallback": fallback,
                "latency_ms": round((time.perf_counter() - started) * 1000, 3)}

    return app
