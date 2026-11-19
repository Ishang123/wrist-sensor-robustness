"""
Steps 4-7 in one pass: corruption robustness + validator gating.

For each subject-wise fold (models from run_baseline.py, never trained on the
test subjects):
  1. fit a WindowValidator on that fold's clean *training* windows only
  2. take a seeded sample of the fold's test windows
  3. for every corruption x severity level (level 0 = clean), corrupt the
     sample, compute features once, and record per window: both models'
     predictions and confidence, and the validator's score/flag/reasons

Everything downstream (tables, plots, gating trade-offs) is computed from the
saved per-window records in results/robustness_records.csv.gz, so analysis
can be re-run without re-running models.
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wrist_sensor_robustness import config, corruptions as C  # noqa: E402
from wrist_sensor_robustness.data import load_windows_cache  # noqa: E402
from wrist_sensor_robustness.features import extract_features  # noqa: E402
from wrist_sensor_robustness.quality import WindowValidator  # noqa: E402

N_TEST_PER_FOLD = 2000  # seeded subsample per fold keeps the full sweep laptop-sized
N_FOLDS = 5


def main():
    X, y, subjects = load_windows_cache()
    model_dir = config.RESULTS_DIR / "models"
    rng = np.random.default_rng(config.RANDOM_SEED)
    records = []

    for fold in range(N_FOLDS):
        with open(model_dir / f"gbm_features_subject_fold{fold}.pkl", "rb") as fh:
            gbm_bundle = pickle.load(fh)
        with open(model_dir / f"cnn_1d_subject_fold{fold}.pkl", "rb") as fh:
            cnn_bundle = pickle.load(fh)
        assert np.array_equal(gbm_bundle["test_idx"], cnn_bundle["test_idx"]), "folds disagree between models"
        tr, te = gbm_bundle["train_idx"], gbm_bundle["test_idx"]
        gbm, cnn = gbm_bundle["model"], cnn_bundle["model"]

        t0 = time.time()
        validator = WindowValidator().fit(X[tr], subject_ids=subjects[tr])
        with open(model_dir / f"validator_fold{fold}.pkl", "wb") as fh:
            pickle.dump(validator, fh)

        sample = np.sort(rng.choice(te, size=min(N_TEST_PER_FOLD, len(te)), replace=False))
        clean = X[sample]
        print(f"fold {fold}: {len(sample)} test windows from {len(np.unique(subjects[sample]))} subjects "
              f"(validator fit {time.time() - t0:.0f}s)", flush=True)

        for kind, levels in C.SEVERITY_LEVELS.items():
            for level in range(len(levels)):
                seed = config.RANDOM_SEED + 1000 * fold + 10 * list(C.SEVERITY_LEVELS).index(kind) + level
                batch = C.corrupt_batch(clean, kind, level, seed=seed)
                feats = extract_features(batch)

                gbm_proba = gbm.predict_proba_from_features(feats)
                cnn_proba = cnn.predict_proba(batch)
                reports = validator.validate_batch(batch, features=feats)

                records.append(pd.DataFrame({
                    "fold": fold,
                    "window_idx": sample,
                    "subject_id": subjects[sample],
                    "true": y[sample],
                    "corruption": kind,
                    "level": level,
                    "severity": str(levels[level]),
                    "gbm_pred": gbm.classes_[gbm_proba.argmax(1)],
                    "gbm_conf": gbm_proba.max(1),
                    "cnn_pred": cnn.classes_[cnn_proba.argmax(1)],
                    "cnn_conf": cnn_proba.max(1),
                    "quality_score": [r.quality_score for r in reports],
                    "flagged": [r.flagged for r in reports],
                    "reasons": ["|".join(r.reasons) for r in reports],
                }))
            print(f"  {kind:10s} done ({time.time() - t0:.0f}s)", flush=True)

    out = pd.concat(records, ignore_index=True)
    path = config.RESULTS_DIR / "robustness_records.csv.gz"
    out.to_csv(path, index=False)
    print(f"\n{len(out):,} records -> {path}")


if __name__ == "__main__":
    main()
