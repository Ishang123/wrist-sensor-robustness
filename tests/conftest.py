import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wrist_sensor_robustness import config  # noqa: E402

REAL_DATA = config.RAW_WATCH_ACCEL_DIR.exists() and any(config.RAW_WATCH_ACCEL_DIR.glob("*.txt"))
requires_real_data = pytest.mark.skipif(not REAL_DATA, reason="WISDM raw data not present in data/raw/watch")


def synthetic_window(rng, freq_hz=1.5, amp=3.0, noise=0.3, n=config.WINDOW_SAMPLES, fs=config.NOMINAL_SAMPLE_RATE_HZ):
    """Gravity on z plus a periodic motion and sensor noise - shaped like a real watch window."""
    t = np.arange(n) / fs
    phase = rng.uniform(0, 2 * np.pi)
    w = np.zeros((n, 3))
    w[:, 0] = amp * np.sin(2 * np.pi * freq_hz * t + phase)
    w[:, 1] = 0.5 * amp * np.cos(2 * np.pi * freq_hz * t + phase)
    w[:, 2] = 9.81
    return w + rng.normal(0, noise, size=w.shape)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def clean_windows(rng):
    """200 synthetic windows from two 'activities' that differ in movement frequency."""
    X = np.stack([synthetic_window(rng, freq_hz=1.0 if i % 2 else 3.0) for i in range(200)])
    y = np.array(["A" if i % 2 else "B" for i in range(200)])
    subjects = np.repeat(np.arange(10), 20)
    return X, y, subjects
