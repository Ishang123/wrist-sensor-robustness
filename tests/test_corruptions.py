import numpy as np
import pytest

from conftest import synthetic_window
from wrist_sensor_robustness import corruptions as C
from wrist_sensor_robustness.quality import clip_fraction, linear_run


@pytest.fixture
def window(rng):
    return synthetic_window(rng)


@pytest.mark.parametrize("kind", list(C.SEVERITY_LEVELS))
def test_level_zero_is_identity_and_input_untouched(kind, window, rng):
    before = window.copy()
    out = C.corrupt(window, kind, 0, rng)
    assert np.array_equal(out, window)
    assert np.array_equal(window, before)


@pytest.mark.parametrize("kind", list(C.SEVERITY_LEVELS))
@pytest.mark.parametrize("level", [1, 3, 5])
def test_shape_finite_and_input_not_mutated(kind, level, window, rng):
    before = window.copy()
    out = C.corrupt(window, kind, level, rng)
    assert out.shape == window.shape and np.isfinite(out).all()
    assert np.array_equal(window, before)


@pytest.mark.parametrize("kind", list(C.SEVERITY_LEVELS))
def test_batches_are_reproducible_from_seed(kind, rng):
    X = np.stack([synthetic_window(rng) for _ in range(5)])
    assert np.array_equal(C.corrupt_batch(X, kind, 4, seed=7), C.corrupt_batch(X, kind, 4, seed=7))


def test_severity_increases_distortion_on_average(rng):
    X = np.stack([synthetic_window(rng, amp=6) for _ in range(30)])
    # rate_drift is excluded: pointwise error between a sine and its time-stretched copy
    # saturates once they drift out of phase; its severity is tested via frequency below.
    for kind in ("dropout", "noise", "rotation", "clipping"):
        deltas = [np.abs(C.corrupt_batch(X, kind, lvl, seed=lvl) - X).mean() for lvl in range(6)]
        assert deltas == sorted(deltas), (kind, deltas)


@pytest.mark.parametrize("level", [1, 2, 3, 4, 5])
def test_dropout_loses_exactly_the_requested_fraction(level, rng):
    w = synthetic_window(rng)
    out = C.corrupt(w, "dropout", level, rng)
    changed = np.any(np.abs(out - w) > 1e-12, axis=1)
    expected = round(C.SEVERITY_LEVELS["dropout"][level] * len(w))
    # every dropped sample is replaced by interpolation, so (almost surely) changed
    assert changed.sum() == expected


def test_rotation_preserves_magnitude(window, rng):
    out = C.corrupt(window, "rotation", 5, rng)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), np.linalg.norm(window, axis=1))


def test_axis_swap_is_signed_permutation(window, rng):
    out = C.corrupt(window, "axis_swap", 2, rng)  # swap_xy
    np.testing.assert_array_equal(out[:, 0], window[:, 1])
    np.testing.assert_array_equal(out[:, 1], window[:, 0])
    np.testing.assert_array_equal(np.sort(np.abs(out), axis=None), np.sort(np.abs(window), axis=None))


def test_clipping_bounds_values(window, rng):
    limit = C.SEVERITY_LEVELS["clipping"][5]
    out = C.corrupt(window * 3, "clipping", 5, rng)
    assert np.abs(out).max() <= limit
    assert clip_fraction(out) > 0.1


def test_dropout_leaves_linear_interpolation_ramps(window, rng):
    assert linear_run(window) < 3
    assert linear_run(C.corrupt(window, "dropout", 3, rng)) >= 10


def test_invalid_kind_or_level_raises(window, rng):
    with pytest.raises(ValueError):
        C.corrupt(window, "gremlins", 1, rng)
    with pytest.raises(ValueError):
        C.corrupt(window, "noise", 6, rng)


@pytest.mark.parametrize("level", [2, 4, 5])
def test_rate_drift_slows_motion_by_expected_factor(level, rng):
    fs, f0 = 20, 2.0
    t = np.arange(400) / fs
    w = np.column_stack([np.sin(2 * np.pi * f0 * t), np.zeros_like(t), np.full_like(t, 9.81)])
    out = C.corrupt(w, "rate_drift", level, rng)
    spec = np.abs(np.fft.rfft(out[:, 0] - out[:, 0].mean()))
    freqs = np.fft.rfftfreq(len(t), 1 / fs)
    expected = f0 / (1 + C.SEVERITY_LEVELS["rate_drift"][level])
    assert abs(freqs[spec.argmax()] - expected) <= freqs[1]  # within one FFT bin
