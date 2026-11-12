"""Inference-service tests (skipped automatically when fastapi/httpx or examples/ are not installed)."""
import time

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

create_app = pytest.importorskip(
    "serving_app_example",
    reason="examples/ is not installed: copy it next to tests/ to run the serving tests").create_app

SCHEMA = {"columns": {
    "size": {"dtype": "number", "nullable": False, "min": 20, "max": 5000},
    "rooms": {"dtype": "number", "nullable": False, "min": 1, "max": 30},
    "district": {"dtype": "str", "nullable": False, "allowed": ["A", "B", "C", "D"]}}}
GOOD = {"size": 250.0, "rooms": 5, "district": "B"}


@pytest.fixture
def client(hooks, require):
    require("load_pipeline")
    return TestClient(create_app(hooks.load_pipeline(), SCHEMA, model_version="test-1", fallback_value=None))


def test_health_reports_version(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok", "model_version": "test-1"}


def test_predict_matches_direct_pipeline_call(hooks, client):
    recs = [GOOD, {"size": 120.0, "rooms": 3, "district": "A"}]
    body = client.post("/predict", json={"records": recs}).json()
    direct = hooks.load_pipeline().predict(pd.DataFrame(recs))
    assert np.allclose(body["predictions"], direct) and body["fallback"] is False
    assert body["model_version"] == "test-1"


@pytest.mark.parametrize("payload", [
    {}, {"records": []}, {"records": "x"},
    {"records": [{"size": 250.0, "rooms": 5}]},                      # missing column
    {"records": [{**GOOD, "district": "ZZ"}]},                       # unseen category
    {"records": [{**GOOD, "size": 99999}]},                          # out of range
    {"records": [{**GOOD, "size": "big"}]},                          # wrong type
])
def test_invalid_requests_are_rejected_with_422(client, payload):
    r = client.post("/predict", json=payload)
    assert r.status_code == 422 and r.json()["detail"]


def test_batch_limit_returns_413(hooks, require):
    require("load_pipeline")
    small = TestClient(create_app(hooks.load_pipeline(), SCHEMA, "t", max_batch=2))
    assert small.post("/predict", json={"records": [GOOD] * 3}).status_code == 413


class _Broken:
    def predict(self, X):
        raise RuntimeError("boom")


def test_model_failure_uses_fallback_when_configured_else_503():
    with_fb = TestClient(create_app(_Broken(), SCHEMA, "t", fallback_value=123.0))
    body = with_fb.post("/predict", json={"records": [GOOD, GOOD]}).json()
    assert body["fallback"] is True and body["predictions"] == [123.0, 123.0]
    no_fb = TestClient(create_app(_Broken(), SCHEMA, "t"))
    assert no_fb.post("/predict", json={"records": [GOOD]}).status_code == 503


def test_latency_budget_for_a_small_batch(client):
    recs = [GOOD] * 50
    client.post("/predict", json={"records": recs})  # warm-up
    times = []
    for _ in range(30):
        t = time.perf_counter()
        assert client.post("/predict", json={"records": recs}).status_code == 200
        times.append(time.perf_counter() - t)
    p95 = float(np.quantile(times, 0.95))
    assert p95 < 0.5, f"p95 latency {p95 * 1000:.0f} ms exceeds the 500 ms test budget"
