"""
Cross-validation splits and metrics.

Two split strategies, deliberately side by side:
  - subject_splits: GroupKFold by subject. Every window from a given person is
    in exactly one fold, so test subjects are never seen in training. This is
    the honest estimate of performance on a new wearer, and the only one
    reported as the result.
  - random_window_splits: StratifiedKFold over windows, ignoring subject.
    Windows from the same person (and, with 50% overlap, windows that share
    half their samples) land on both sides of the split. Reported only to
    quantify the leakage inflation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from . import config


def subject_splits(subject_ids: np.ndarray, n_splits: int = 5, seed: int = config.RANDOM_SEED):
    """GroupKFold with subjects shuffled first, so fold membership is seeded rather than ID-ordered."""
    rng = np.random.default_rng(seed)
    unique = np.unique(subject_ids)
    perm = dict(zip(unique, rng.permutation(len(unique))))
    shuffled_groups = np.array([perm[s] for s in subject_ids])
    yield from GroupKFold(n_splits=n_splits).split(np.zeros(len(subject_ids)), groups=shuffled_groups)


def random_window_splits(labels: np.ndarray, n_splits: int = 5, seed: int = config.RANDOM_SEED):
    yield from StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(
        np.zeros(len(labels)), labels
    )


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


@dataclass
class FoldResult:
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    model: object
    metrics: dict[str, float]
    test_subjects: list[int] = field(default_factory=list)


def cross_validate(model_factory, X, y, splits, subject_ids=None, verbose=True) -> list[FoldResult]:
    """Train a fresh model per fold, score on the held-out fold, keep the fitted models for reuse."""
    results = []
    for fold, (tr, te) in enumerate(splits):
        model = model_factory().fit(X[tr], y[tr])
        metrics = score(y[te], model.predict(X[te]))
        subs = sorted(set(subject_ids[te].tolist())) if subject_ids is not None else []
        results.append(FoldResult(fold, tr, te, model, metrics, subs))
        if verbose:
            print(f"  fold {fold}: acc={metrics['accuracy']:.3f} macro_f1={metrics['macro_f1']:.3f}"
                  + (f" ({len(subs)} test subjects)" if subs else ""), flush=True)
    return results


def summarize(results: list[FoldResult]) -> dict[str, float]:
    out = {}
    for key in ("accuracy", "macro_f1"):
        vals = np.array([r.metrics[key] for r in results])
        out[f"{key}_mean"] = float(vals.mean())
        out[f"{key}_std"] = float(vals.std())
    return out
