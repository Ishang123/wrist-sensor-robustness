"""
FastAPI service: validate a window of smartwatch accelerometer data, then
(only if it passes) classify the activity.

    POST /predict
        {
          "samples": [[x, y, z], ...],     # 200 rows = 10 s at 20 Hz
          "timestamps_ns": [...],          # optional, enables a direct gap check
          "force": false                   # run the model even if flagged
        }
    ->  {
          "prediction": "walking" | null,  # null when flagged and not forced
          "activity_code": "A" | null,
          "probabilities": {"walking": 0.91, ...},   # top 3
          "quality_score": 0.0-1.0,
          "flagged": true/false,
          "reasons": ["gap", ...],
          "checks": {...},                 # raw values behind each rule
          "model_run": true/false
        }

    GET /health  -> model + validator metadata

Artifacts are loaded from results/models/final_*.pkl (scripts/train_final.py).
The path can be overridden with WSR_MODEL_DIR, which the tests use to inject
small, quickly trained artifacts.
"""

from __future__ import annotations

import os
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from . import config
from .data import load_activity_key


class PredictRequest(BaseModel):
    samples: list[list[float]] = Field(..., description="window of [x, y, z] accelerometer rows in m/s^2")
    timestamps_ns: list[int] | None = Field(None, description="optional per-sample timestamps (ns)")
    force: bool = Field(False, description="run the model even if the window is flagged")

    @field_validator("samples")
    @classmethod
    def _shape(cls, v):
        if len(v) != config.WINDOW_SAMPLES:
            raise ValueError(f"expected {config.WINDOW_SAMPLES} samples "
                             f"({config.WINDOW_SECONDS} s at {config.NOMINAL_SAMPLE_RATE_HZ} Hz), got {len(v)}")
        if any(len(row) != 3 for row in v):
            raise ValueError("every sample must be [x, y, z]")
        if not np.isfinite(np.asarray(v, dtype=float)).all():
            raise ValueError("samples contain NaN or infinite values")
        return v

    @field_validator("timestamps_ns")
    @classmethod
    def _ts(cls, v, info):
        if v is not None and len(v) != config.WINDOW_SAMPLES:
            raise ValueError("timestamps_ns must have one entry per sample")
        return v


def _model_dir() -> Path:
    return Path(os.environ.get("WSR_MODEL_DIR", config.RESULTS_DIR / "models"))


@lru_cache(maxsize=1)
def load_artifacts():
    d = _model_dir()
    try:
        with open(d / "final_gbm_features.pkl", "rb") as fh:
            model = pickle.load(fh)
        with open(d / "final_validator.pkl", "rb") as fh:
            validator = pickle.load(fh)
    except FileNotFoundError as e:
        raise RuntimeError(f"missing model artifact in {d}; run `make train-final` first") from e
    try:
        names = load_activity_key()
    except FileNotFoundError:
        names = {}
    return model, validator, names


app = FastAPI(title="wrist-sensor-robustness", version="0.1.0")


@app.get("/health")
def health():
    model, validator, _ = load_artifacts()
    return {"status": "ok", "model": type(model).__name__, "n_classes": len(model.classes_),
            "validator_alpha": validator.alpha, "window_samples": config.WINDOW_SAMPLES,
            "sample_rate_hz": config.NOMINAL_SAMPLE_RATE_HZ}


@app.post("/predict")
def predict(req: PredictRequest):
    model, validator, names = load_artifacts()
    window = np.asarray(req.samples, dtype=float)
    ts = None if req.timestamps_ns is None else np.asarray(req.timestamps_ns, dtype=np.int64)
    if ts is not None and np.any(np.diff(ts) <= 0):
        raise HTTPException(status_code=422, detail="timestamps_ns must be strictly increasing")

    report = validator.validate(window, timestamps_ns=ts)
    response = {**report.to_dict(), "prediction": None, "activity_code": None,
                "probabilities": None, "model_run": False}

    if report.flagged and not req.force:
        return response

    proba = model.predict_proba(window[None])[0]
    order = np.argsort(proba)[::-1]
    code = str(model.classes_[order[0]])
    response.update({
        "prediction": names.get(code, code),
        "activity_code": code,
        "probabilities": {names.get(str(model.classes_[i]), str(model.classes_[i])): round(float(proba[i]), 4)
                          for i in order[:3]},
        "model_run": True,
    })
    return response
