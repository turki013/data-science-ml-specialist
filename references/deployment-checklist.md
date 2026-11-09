# Deployment Readiness Checklist

Use before any model leaves the notebook. Every item is either checked with evidence (command output, test name, file) or marked "not done" with the reason. A model with unchecked **[blocker]** items is not ready.

## 1. One artifact
- [ ] **[blocker]** Preprocessing and model are saved together as ONE object (`Pipeline`), not a model plus loose scalers/encoders.
- [ ] Metadata saved next to it: feature names and dtypes, schema/contract version, library versions, seed, training data period and hash, decision threshold, metrics with intervals.
- [ ] Serialization is `joblib`/pickle only for trusted internal use (loading executes code). For sharing consider `skops` or ONNX. Binaries are not committed to public repositories.
- [ ] Save/load round trip leaves predictions identical (`parity.assert_roundtrip_equal`).

## 2. Train/serve parity
- [ ] **[blocker]** One feature implementation, or a parity test between the training path and the serving path (`parity.assert_feature_parity`).
- [ ] Batch predictions equal single-record predictions (`assert_batch_equals_rowwise`).
- [ ] Predictions do not depend on input column order (`assert_column_order_invariant`) and are deterministic (`assert_deterministic`).
- [ ] The serving path uses the same text fields, normalizers and defaults as training.

```bash
pytest tests/test_pipeline_parity.py tests/test_model_regression.py tests/test_data_leakage.py
```

## 3. Input validation
- [ ] **[blocker]** Every request is validated against the data contract (`validate_records`): required fields, types, ranges, allowed categories. Invalid input returns HTTP 422 with the list of problems.
- [ ] No silent defaults for missing fields (a missing value is rejected or imputed by the pipeline's own fitted imputer).
- [ ] Unseen categories are handled explicitly (rejected, or mapped to "other" by design).
- [ ] Batch size limit (HTTP 413) and request timeout.

## 4. Inference service (FastAPI pattern)
`create_app(pipeline, schema, model_version, fallback_value=None)` validates, predicts, labels the response with the model version, and falls back (flagged) or returns 503 when the model fails. Tested in `tests/test_serving_api.py`.

```python
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
```

Run: `uvicorn app:app --host 0.0.0.0 --port 8000` after wiring your own pipeline and schema in `app.py`.

## 5. Latency and capacity
- [ ] Measure on production-like hardware, warm and cold: p50, p95, p99 for the real batch sizes.
- [ ] A latency budget is written down (e.g., p95 under 200 ms for 1 record) and a test enforces it.
- [ ] Memory footprint and cold-start time are known.

```python
def measure_latency(fn: Callable[[], object], n: int = 100, warmup: int = 5) -> Dict[str, float]:
    """Call fn() warmup + n times; return p50/p95/p99/mean/max in milliseconds (warm-up excluded)."""
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(n):
        s = time.perf_counter()
        fn()
        t.append((time.perf_counter() - s) * 1000.0)
    a = np.asarray(t)
    return {"n": float(n), "p50_ms": float(np.quantile(a, 0.5)), "p95_ms": float(np.quantile(a, 0.95)),
            "p99_ms": float(np.quantile(a, 0.99)), "mean_ms": float(a.mean()), "max_ms": float(a.max())}
```
```python
from dstk.latency import measure_latency
sample = raw_df.head(1)
print(measure_latency(lambda: pipeline.predict(sample), n=200))     # p50/p95/p99 in ms
```

## 6. Fallbacks and failure behavior
- [ ] Defined behavior when: model file missing or corrupt, feature extraction raises, dependency unavailable, input out of range.
- [ ] Fallback value or rule is documented, flagged in the response (`"fallback": true`) and counted in monitoring.
- [ ] The service never returns a confident number for data the contract rejects.

## 7. Release and rollback
- [ ] Model version in every response and log line; previous version kept for instant rollback.
- [ ] Shadow mode or canary before full rollout; success and abort criteria written.
- [ ] Reproducible build: pinned `requirements.txt`, code hash, data hash, seed.

## 8. Monitoring plan
- [ ] Input drift: PSI/KS per feature against the training reference (`dstk.drift`); alert thresholds set (for example PSI above 0.25).
- [ ] Prediction distribution drift and error rate; when labels arrive late, rolling error by slice.
- [ ] Data-contract violations counted per field.
- [ ] Retrain trigger and owner defined; logs contain no personal data beyond what is necessary.

## 9. Documentation and risk
- [ ] Model Card completed (limitations and worst slices first).
- [ ] README states what the model predicts, what it does not, and the data period it reflects.
- [ ] Security: no secrets in the repo or image, dependency scan, request size limits, authentication where required.
- [ ] Fairness and privacy review done if the output affects people.

## 10. Final gate
- [ ] Full test suite green (`pytest`), including the frozen-set regression test.
- [ ] The frozen test set matches its **hardcoded SHA256 literal** (`canonical_fingerprint`); the expected value is not recomputed from the same file.
- [ ] `python scripts/check_deliverables.py` prints OK (reports in the repository, results integrity, limitations first).
- [ ] `python scripts/lint_report.py README.md --results reports/results.json` passes.
- [ ] Someone other than the author can reproduce the reported numbers from the README.
