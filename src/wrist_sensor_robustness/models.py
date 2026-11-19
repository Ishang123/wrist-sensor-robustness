"""
Model wrappers with a uniform fit(windows, labels) / predict(windows) API.

Both models take raw windows of shape (N, n_samples, 3), so the robustness
study and the API can corrupt or validate windows once and hand them to
either model unchanged:
  - FeatureGBM: hand-crafted features (features.py) -> XGBoost
  - CNNModel:   per-channel standardization -> SmallCNN1D (cnn.py)

Any statistic computed from data (label encoding, standardization mean/std)
is fit on the training fold only, so nothing about held-out subjects leaks
into training.
"""

from __future__ import annotations

import numpy as np
from xgboost import XGBClassifier

from . import config
from .cnn import SmallCNN1D
from .features import extract_features


class FeatureGBM:
    name = "gbm_features"

    def __init__(self, seed: int = config.RANDOM_SEED, n_estimators: int = 300):
        self.seed = seed
        self.n_estimators = n_estimators

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FeatureGBM":
        self.classes_ = np.unique(y)
        idx = np.searchsorted(self.classes_, y)
        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=self.seed,
            n_jobs=-1,
            verbosity=0,
        )
        self.model.fit(extract_features(X), idx)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba_from_features(extract_features(X))

    def predict_proba_from_features(self, feats: np.ndarray) -> np.ndarray:
        """Skip feature extraction when the caller already has features (shared with the validator)."""
        return self.model.predict_proba(feats)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class CNNModel:
    name = "cnn_1d"

    def __init__(self, seed: int = config.RANDOM_SEED, n_epochs: int = 20,
                 conv1_filters: int = 16, conv2_filters: int = 32, conv3_filters: int = 64,
                 kernel_size: int = 7,
                 learning_rate: float = 3e-3):
        self.kw = dict(n_epochs=n_epochs, conv1_filters=conv1_filters, conv2_filters=conv2_filters,
                       conv3_filters=conv3_filters,
                       kernel_size=kernel_size, learning_rate=learning_rate, seed=seed)

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean_) / self.std_).astype(np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CNNModel":
        self.mean_ = X.mean(axis=(0, 1))
        self.std_ = X.std(axis=(0, 1)) + 1e-8
        self.net = SmallCNN1D(n_channels=X.shape[2], n_classes=len(np.unique(y)),
                              window_length=X.shape[1], **self.kw)
        self.net.fit(self._standardize(X), y)
        self.classes_ = self.net.classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Batch to bound memory on large evaluation sets.
        Xs = self._standardize(X)
        return np.vstack([self.net.predict_proba(Xs[i:i + 2048]) for i in range(0, len(Xs), 2048)])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


MODEL_FACTORIES = {
    FeatureGBM.name: FeatureGBM,
    CNNModel.name: CNNModel,
}
