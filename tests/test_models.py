import numpy as np

from wrist_sensor_robustness import features as F
from wrist_sensor_robustness import corruptions as C
from wrist_sensor_robustness.cnn import SmallCNN1D, softmax_cross_entropy
from wrist_sensor_robustness.evaluation import random_window_splits, subject_splits
from wrist_sensor_robustness.models import CNNModel, FeatureGBM


def test_cnn_gradients_match_finite_differences():
    """Hand-written backprop must agree with numerical gradients for every layer's parameters."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 40, 3))
    y_idx = rng.integers(0, 4, size=6)
    net = SmallCNN1D(n_channels=3, n_classes=4, window_length=40, conv1_filters=4,
                     conv2_filters=4, conv3_filters=4, kernel_size=3, stride=2, seed=0)

    def loss():
        return softmax_cross_entropy(net._forward(X), y_idx)

    _, d_logits = loss()
    net._backward(d_logits)
    eps, worst = 1e-5, 0.0
    for layer in net._layers:
        for p, g in layer.params_and_grads():
            flat_p, flat_g = p.reshape(-1), g.reshape(-1)
            for i in rng.choice(flat_p.size, size=min(5, flat_p.size), replace=False):
                orig = flat_p[i]
                flat_p[i] = orig + eps
                lp, _ = loss()
                flat_p[i] = orig - eps
                lm, _ = loss()
                flat_p[i] = orig
                num = (lp - lm) / (2 * eps)
                worst = max(worst, abs(num - flat_g[i]) / max(abs(num) + abs(flat_g[i]), 1e-8))
    assert worst < 1e-4


def test_cnn_learns_separable_task(clean_windows):
    X, y, _ = clean_windows
    m = CNNModel(n_epochs=15, conv1_filters=8, conv2_filters=8, conv3_filters=8).fit(X, y)
    assert (m.predict(X) == y).mean() > 0.9


def test_gbm_learns_separable_task(clean_windows):
    X, y, _ = clean_windows
    m = FeatureGBM(n_estimators=20).fit(X, y)
    assert (m.predict(X) == y).mean() > 0.9
    assert m.predict_proba(X).shape == (len(X), 2)


def test_feature_vector_matches_names(clean_windows):
    X, _, _ = clean_windows
    assert F.extract_features(X[:3]).shape == (3, len(F.feature_names()))


def test_magnitude_features_invariant_to_orientation(clean_windows, rng):
    X, _, _ = clean_windows
    names = F.feature_names()
    mag_cols = [i for i, n in enumerate(names) if n.startswith("mag_")]
    rotated = C.corrupt(X[0], "rotation", 5, rng)
    np.testing.assert_allclose(F.extract_window_features(X[0])[mag_cols],
                               F.extract_window_features(rotated)[mag_cols], rtol=1e-6, atol=1e-6)


def test_subject_splits_never_share_a_subject():
    subjects = np.repeat(np.arange(20), 15)
    for tr, te in subject_splits(subjects, n_splits=5):
        assert not set(subjects[tr]) & set(subjects[te])


def test_random_window_split_does_leak_subjects():
    """The comparison split is supposed to mix subjects - that's the point of reporting it."""
    subjects = np.repeat(np.arange(20), 15)
    labels = np.tile(np.arange(3), 100)
    tr, te = next(random_window_splits(labels, n_splits=5))
    assert set(subjects[tr]) & set(subjects[te])
