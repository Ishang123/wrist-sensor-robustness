"""
Train the artifacts the API serves: the feature GBM and the validator, fit on
all 51 subjects. Reported performance comes from subject-wise CV
(run_baseline.py), not from these models, which have no held-out data left.
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wrist_sensor_robustness import config  # noqa: E402
from wrist_sensor_robustness.data import load_windows_cache  # noqa: E402
from wrist_sensor_robustness.models import FeatureGBM  # noqa: E402
from wrist_sensor_robustness.quality import WindowValidator  # noqa: E402


def main():
    X, y, subjects = load_windows_cache()
    out = config.RESULTS_DIR / "models"
    out.mkdir(parents=True, exist_ok=True)
    model = FeatureGBM().fit(X, y)
    validator = WindowValidator().fit(X, subject_ids=subjects)
    with open(out / "final_gbm_features.pkl", "wb") as fh:
        pickle.dump(model, fh)
    with open(out / "final_validator.pkl", "wb") as fh:
        pickle.dump(validator, fh)
    print(f"trained on {len(X):,} windows from {len(set(subjects))} subjects -> {out}")


if __name__ == "__main__":
    main()
