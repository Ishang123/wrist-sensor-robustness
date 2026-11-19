import numpy as np
import pytest

from conftest import synthetic_window
from wrist_sensor_robustness import corruptions as C
from wrist_sensor_robustness.quality import WindowValidator, clip_fraction, flatline_run, linear_run


@pytest.fixture
def validator(rng):
    X = np.stack([synthetic_window(rng, freq_hz=rng.uniform(0.5, 3)) for _ in range(600)])
    return WindowValidator(alpha=0.01).fit(X, subject_ids=np.repeat(np.arange(30), 20))


def test_rule_primitives(rng):
    w = synthetic_window(rng)
    assert flatline_run(w) == 0 and linear_run(w) < 3 and clip_fraction(w) < 0.05
    w[50:80] = w[50]
    assert flatline_run(w) >= 29


def test_clean_windows_mostly_pass_and_scores_in_unit_interval(validator, rng):
    X = np.stack([synthetic_window(rng, freq_hz=rng.uniform(0.5, 3)) for _ in range(300)])
    reports = validator.validate_batch(X)
    assert np.mean([r.flagged for r in reports]) < 0.05
    assert all(0.0 <= r.quality_score <= 1.0 for r in reports)


def test_flatline_is_flagged(validator, rng):
    w = synthetic_window(rng)
    w[60:120] = w[60]
    r = validator.validate(w)
    assert r.flagged and "flatline" in r.reasons and r.quality_score == 0.0


def test_dropout_gap_is_flagged(validator, rng):
    r = validator.validate(C.corrupt(synthetic_window(rng), "dropout", 3, rng))
    assert r.flagged and "gap" in r.reasons


def test_clipping_is_flagged(validator, rng):
    r = validator.validate(C.corrupt(synthetic_window(rng, amp=8), "clipping", 5, rng))
    assert r.flagged and "clipping" in r.reasons


def test_timestamp_gap_is_flagged_when_timestamps_given(validator, rng):
    ts = np.arange(200, dtype=np.int64) * 50_000_000
    ts[100:] += 2_000_000_000  # 2 s gap
    r = validator.validate(synthetic_window(rng), timestamps_ns=ts)
    assert "timestamp_gap" in r.reasons


def test_heavy_noise_trips_learned_anomaly_score(validator, rng):
    r = validator.validate(C.corrupt(synthetic_window(rng), "noise", 5, rng))
    assert r.flagged


def test_moderate_noise_on_low_motion_window_is_flagged_as_noise(validator, rng):
    w = synthetic_window(rng, amp=0.3, noise=0.05)  # a near-still wrist
    r = validator.validate(w + rng.normal(0, 1.0, size=w.shape))
    assert "noise" in r.reasons


def test_noise_rule_does_not_reject_vigorous_clean_motion(rng):
    """High-motion windows carry high-frequency energy legitimately; the threshold is motion-conditioned."""
    train = np.stack([synthetic_window(rng, freq_hz=rng.uniform(0.5, 4), amp=rng.uniform(0.3, 8),
                                       noise=0.05 + 0.1 * rng.random()) for _ in range(800)])
    v = WindowValidator().fit(train, subject_ids=np.repeat(np.arange(40), 20))
    vigorous = np.stack([synthetic_window(rng, freq_hz=3.5, amp=7.5, noise=0.12) for _ in range(100)])
    assert np.mean(["noise" in r.reasons for r in v.validate_batch(vigorous)]) < 0.05
