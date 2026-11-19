"""
Simulated real-world wearable sensor failures, each with an adjustable severity.

Every corruption takes a single window (n_samples, 3), a severity *level*
(0 = clean, 1..5 = increasingly severe) and a numpy Generator, and returns a
new array of the same shape. Inputs are never modified in place. Levels map to
physical parameters through SEVERITY_LEVELS, so tables and plots can show
both "level 3" and what that means (e.g. "30% of samples dropped").

Corruptions:
  dropout       contiguous bursts of lost samples, covering a given fraction
                of the window, filled by linear interpolation - which is
                exactly what this pipeline's resampler does across a real
                timestamp gap, so the model sees what it would see in
                deployment
  noise         additive Gaussian noise, sigma in m/s^2
  axis_swap     strap/wrist orientation changes as discrete axis permutations
                and sign flips (e.g. watch worn on the other wrist, or rotated
                on the strap)
  rotation      continuous version of the above: rotate the 3-axis signal by
                a given angle about a random axis
  rate_drift    device clock runs fast by a given fraction: the window still
                holds n samples labelled as 20 Hz, but they span less real
                time, so motion appears slowed down by (1 + d)
  clipping      sensor saturation at +/- a given limit in m/s^2

Rate-drift limitation: only the "clock runs fast" direction is simulated.
The opposite direction (window spans *more* real time than labelled) needs
samples beyond the window's end, which a single-window corruption doesn't
have; padding them would add an artifact of the simulator, not of the sensor.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# Level -> physical parameter. Level 0 is always "no corruption".
SEVERITY_LEVELS: dict[str, list] = {
    "dropout": [0.0, 0.1, 0.2, 0.3, 0.5, 0.7],          # fraction of samples lost
    "noise": [0.0, 0.5, 1.0, 2.0, 4.0, 8.0],            # Gaussian sigma, m/s^2
    "axis_swap": [None, "flip_x", "swap_xy", "swap_xy_flip_z", "cycle_xyz", "flip_xyz"],
    "rotation": [0.0, 15.0, 30.0, 45.0, 90.0, 180.0],   # degrees, random axis
    "rate_drift": [0.0, 0.05, 0.10, 0.20, 0.35, 0.50],  # clock runs fast by this fraction
    "clipping": [None, 16.0, 12.0, 8.0, 6.0, 4.0],      # saturation limit, m/s^2 (~1.6g .. 0.4g)
}

SEVERITY_UNITS = {
    "dropout": "fraction of samples lost",
    "noise": "noise sigma (m/s^2)",
    "axis_swap": "axis permutation / sign flip",
    "rotation": "rotation angle (deg)",
    "rate_drift": "clock fast by (fraction)",
    "clipping": "saturation limit (m/s^2)",
}

AXIS_TRANSFORMS = {
    # (new column order, sign per output column)
    "flip_x": ((0, 1, 2), (-1, 1, 1)),
    "swap_xy": ((1, 0, 2), (1, 1, 1)),
    "swap_xy_flip_z": ((1, 0, 2), (1, 1, -1)),
    "cycle_xyz": ((1, 2, 0), (1, 1, 1)),
    "flip_xyz": ((0, 1, 2), (-1, -1, -1)),
}


def dropout(window: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    out = np.array(window, dtype=float, copy=True)
    n = len(out)
    n_drop = int(round(fraction * n))
    if n_drop == 0:
        return out
    # Lose samples in a few contiguous bursts (BLE/firmware drops come in bursts, not
    # independent single samples). Bursts are laid out without overlap so exactly
    # n_drop samples are lost, and the first/last samples are kept so interpolation
    # is defined at both ends.
    n_drop = min(n_drop, n - 2)
    n_bursts = max(1, n_drop // 20)
    burst_sizes = rng.multinomial(n_drop, np.ones(n_bursts) / n_bursts)
    burst_sizes = burst_sizes[burst_sizes > 0]
    n_kept_gaps = len(burst_sizes) + 1
    kept_sizes = rng.multinomial(n - n_drop - 2, np.ones(n_kept_gaps) / n_kept_gaps)
    kept_sizes[0] += 1
    kept_sizes[-1] += 1
    keep = np.ones(n, dtype=bool)
    pos = 0
    for k_size, d_size in zip(kept_sizes[:-1], burst_sizes):
        pos += k_size
        keep[pos:pos + d_size] = False
        pos += d_size
    t = np.arange(n)
    for c in range(out.shape[1]):
        out[:, c] = np.interp(t, t[keep], out[keep, c])
    return out


def noise(window: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return np.asarray(window, dtype=float) + rng.normal(0.0, sigma, size=np.shape(window))


def axis_swap(window: np.ndarray, transform: str | None, rng: np.random.Generator) -> np.ndarray:
    out = np.array(window, dtype=float, copy=True)
    if transform is None:
        return out
    order, signs = AXIS_TRANSFORMS[transform]
    return out[:, list(order)] * np.asarray(signs, dtype=float)


def _rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle_rad) * k + (1 - np.cos(angle_rad)) * (k @ k)  # Rodrigues


def rotation(window: np.ndarray, angle_deg: float, rng: np.random.Generator) -> np.ndarray:
    out = np.asarray(window, dtype=float)
    if angle_deg == 0:
        return out.copy()
    axis = rng.normal(size=3)
    return out @ _rotation_matrix(axis, np.deg2rad(angle_deg)).T


def rate_drift(window: np.ndarray, drift: float, rng: np.random.Generator) -> np.ndarray:
    out = np.asarray(window, dtype=float)
    if drift == 0:
        return out.copy()
    n = len(out)
    # Sample i of the corrupted window corresponds to original position i / (1 + drift),
    # which always lies inside [0, n-1] for drift > 0 - no extrapolation needed.
    src = np.arange(n) / (1.0 + drift)
    t = np.arange(n)
    return np.column_stack([np.interp(src, t, out[:, c]) for c in range(out.shape[1])])


def clipping(window: np.ndarray, limit: float | None, rng: np.random.Generator) -> np.ndarray:
    out = np.asarray(window, dtype=float)
    if limit is None:
        return out.copy()
    return np.clip(out, -limit, limit)


CORRUPTIONS: dict[str, Callable] = {
    "dropout": dropout,
    "noise": noise,
    "axis_swap": axis_swap,
    "rotation": rotation,
    "rate_drift": rate_drift,
    "clipping": clipping,
}


def corrupt(window: np.ndarray, kind: str, level: int, rng: np.random.Generator) -> np.ndarray:
    """Apply corruption `kind` at severity `level` (0..5) to one window."""
    if kind not in CORRUPTIONS:
        raise ValueError(f"unknown corruption {kind!r}; choose from {sorted(CORRUPTIONS)}")
    levels = SEVERITY_LEVELS[kind]
    if not 0 <= level < len(levels):
        raise ValueError(f"level must be in 0..{len(levels) - 1}, got {level}")
    return CORRUPTIONS[kind](window, levels[level], rng)


def corrupt_batch(windows: np.ndarray, kind: str, level: int, seed: int) -> np.ndarray:
    """Corrupt every window in a batch with an independent but seeded draw."""
    rng = np.random.default_rng(seed)
    return np.stack([corrupt(w, kind, level, rng) for w in windows])
