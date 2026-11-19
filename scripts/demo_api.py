"""
Send real WISDM windows to the API in-process (no server needed) - one clean,
then the same window under several corruptions - and print the responses.

Uses the final artifacts from train_final.py, which were trained on all
subjects, so this demonstrates the API's behaviour, not held-out accuracy.
"""

import json
import sys
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wrist_sensor_robustness import corruptions as C  # noqa: E402
from wrist_sensor_robustness.api import app  # noqa: E402
from wrist_sensor_robustness.data import load_windows_cache  # noqa: E402


def main():
    X, y, subjects = load_windows_cache()
    i = int(np.flatnonzero(y == "A")[100])  # a walking window
    client = TestClient(app)
    rng = np.random.default_rng(0)
    cases = {"clean": X[i]}
    for kind, level in (("dropout", 2), ("clipping", 4), ("noise", 5), ("rotation", 4)):
        cases[f"{kind} L{level}"] = C.corrupt(X[i], kind, level, rng)

    print(f"window {i}: subject {subjects[i]}, true activity {y[i]} (walking)\n")
    for name, w in cases.items():
        r = client.post("/predict", json={"samples": w.tolist()}).json()
        short = {k: r[k] for k in ("prediction", "quality_score", "flagged", "reasons", "model_run")}
        print(f"{name:12s} {json.dumps(short)}")


if __name__ == "__main__":
    main()
