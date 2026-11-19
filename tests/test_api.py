import pickle

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import synthetic_window
from wrist_sensor_robustness import api
from wrist_sensor_robustness.models import FeatureGBM
from wrist_sensor_robustness.quality import WindowValidator


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    rng = np.random.default_rng(0)
    X = np.stack([synthetic_window(rng, freq_hz=1.0 if i % 2 else 3.0) for i in range(300)])
    y = np.array(["A" if i % 2 else "B" for i in range(300)])
    d = tmp_path_factory.mktemp("models")
    with open(d / "final_gbm_features.pkl", "wb") as fh:
        pickle.dump(FeatureGBM(n_estimators=20).fit(X, y), fh)
    with open(d / "final_validator.pkl", "wb") as fh:
        pickle.dump(WindowValidator().fit(X, subject_ids=np.repeat(np.arange(15), 20)), fh)
    mp = pytest.MonkeyPatch()
    mp.setenv("WSR_MODEL_DIR", str(d))
    api.load_artifacts.cache_clear()
    yield TestClient(api.app)
    api.load_artifacts.cache_clear()
    mp.undo()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["n_classes"] == 2


def test_clean_window_gets_prediction(client):
    w = synthetic_window(np.random.default_rng(1), freq_hz=1.0)
    body = client.post("/predict", json={"samples": w.tolist()}).json()
    assert body["model_run"] and not body["flagged"]
    assert body["activity_code"] == "A"
    assert 0 < body["quality_score"] <= 1


def test_flagged_window_is_rejected_before_model(client):
    w = synthetic_window(np.random.default_rng(2))
    w[40:140] = w[40]
    body = client.post("/predict", json={"samples": w.tolist()}).json()
    assert body["flagged"] and "flatline" in body["reasons"]
    assert body["prediction"] is None and not body["model_run"]


def test_force_runs_model_on_flagged_window(client):
    w = synthetic_window(np.random.default_rng(2))
    w[40:140] = w[40]
    body = client.post("/predict", json={"samples": w.tolist(), "force": True}).json()
    assert body["flagged"] and body["model_run"] and body["prediction"] is not None


@pytest.mark.parametrize("samples", [
    [[0.0, 0.0, 9.8]] * 50,              # wrong length
    [[0.0, 9.8]] * 200,                  # wrong width
])
def test_malformed_input_is_422(client, samples):
    assert client.post("/predict", json={"samples": samples}).status_code == 422


def test_non_monotonic_timestamps_rejected(client):
    w = synthetic_window(np.random.default_rng(3))
    ts = list(range(200))
    ts[10] = 0
    r = client.post("/predict", json={"samples": w.tolist(), "timestamps_ns": ts})
    assert r.status_code == 422
