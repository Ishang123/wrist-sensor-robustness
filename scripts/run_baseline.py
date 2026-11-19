"""
Step 2-3: train both models with subject-wise 5-fold CV, and with a random
window split for comparison (leakage inflation).

Outputs:
  results/tables/baseline_cv.csv       per-fold metrics, both split types
  results/tables/baseline_summary.csv  mean +/- std per model and split type
  results/models/<model>_subject_fold<k>.pkl  fitted fold models (reused by
      the robustness and validator-gating studies, so every downstream result
      is measured on subjects the model never trained on)
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wrist_sensor_robustness import config, evaluation as ev, models  # noqa: E402
from wrist_sensor_robustness.data import load_windows_cache  # noqa: E402


def main():
    np.random.seed(config.RANDOM_SEED)
    X, y, subjects = load_windows_cache()
    print(f"{len(X):,} windows, {len(np.unique(subjects))} subjects, {len(np.unique(y))} activities")

    model_dir = config.RESULTS_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for split_name in ("subject", "random_window"):
        for model_name, factory in models.MODEL_FACTORIES.items():
            print(f"\n[{split_name}] {model_name}", flush=True)
            splits = (ev.subject_splits(subjects) if split_name == "subject"
                      else ev.random_window_splits(y))
            t0 = time.time()
            results = ev.cross_validate(factory, X, y, splits, subject_ids=subjects)
            print(f"  {time.time() - t0:.0f}s", flush=True)
            for r in results:
                rows.append({"split": split_name, "model": model_name, "fold": r.fold,
                             "n_test_windows": len(r.test_idx), "n_test_subjects": len(r.test_subjects),
                             **r.metrics})
                if split_name == "subject":
                    with open(model_dir / f"{model_name}_subject_fold{r.fold}.pkl", "wb") as fh:
                        pickle.dump({"model": r.model, "train_idx": r.train_idx, "test_idx": r.test_idx,
                                     "test_subjects": r.test_subjects}, fh)

    per_fold = pd.DataFrame(rows)
    tables = config.RESULTS_DIR / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    per_fold.to_csv(tables / "baseline_cv.csv", index=False)

    summary = (per_fold.groupby(["model", "split"])[["accuracy", "macro_f1"]]
               .agg(["mean", "std"]).round(4))
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(tables / "baseline_summary.csv", index=False)
    print("\n", summary.to_string(index=False))


if __name__ == "__main__":
    main()
