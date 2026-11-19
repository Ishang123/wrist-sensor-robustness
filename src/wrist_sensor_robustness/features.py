"""
Hand-crafted window features for the gradient-boosted baseline.

Per channel (x, y, z, and the orientation-invariant magnitude |a|):
  - time domain: mean, std, min, max, 10/50/90th percentiles, mean absolute
    first difference (jerk proxy), zero-crossing rate of the mean-removed signal
  - frequency domain: total spectral energy, energy in 4 bands
    (0-1, 1-3, 3-5, 5-10 Hz), dominant frequency
  - cadence-like: lag of the strongest autocorrelation peak in the 0.25-2.0 s
    range, expressed as a period in seconds, plus that peak's height (how
    periodic the movement is)

Plus cross-axis Pearson correlations (xy, xz, yz).

The magnitude channel matters for the robustness study: it is invariant to
axis swaps and sign flips, so a model leaning on it should degrade less under
strap/orientation corruptions. Whether it actually does is measured, not
assumed.
"""

from __future__ import annotations

import numpy as np

from . import config

FREQ_BANDS_HZ = ((0.0, 1.0), (1.0, 3.0), (3.0, 5.0), (5.0, 10.0))
CADENCE_MIN_S, CADENCE_MAX_S = 0.25, 2.0


def _channel_features(sig: np.ndarray, fs: float) -> list[float]:
    centered = sig - sig.mean()
    feats = [
        sig.mean(),
        sig.std(),
        sig.min(),
        sig.max(),
        *np.percentile(sig, [10, 50, 90]),
        np.abs(np.diff(sig)).mean(),
        np.mean(np.diff(np.signbit(centered).astype(np.int8)) != 0),
    ]

    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    freqs = np.fft.rfftfreq(len(sig), d=1.0 / fs)
    total = spectrum.sum()
    feats.append(total / len(sig))
    for lo, hi in FREQ_BANDS_HZ:
        band = spectrum[(freqs >= lo) & (freqs < hi)].sum()
        feats.append(band / total if total > 0 else 0.0)
    feats.append(freqs[np.argmax(spectrum[1:]) + 1] if len(spectrum) > 1 else 0.0)

    # Cadence: autocorrelation peak within a plausible step/stroke period range.
    denom = np.dot(centered, centered)
    lo_lag, hi_lag = int(CADENCE_MIN_S * fs), int(CADENCE_MAX_S * fs)
    if denom > 0 and hi_lag < len(sig):
        ac = np.correlate(centered, centered, mode="full")[len(sig) - 1 :] / denom
        seg = ac[lo_lag : hi_lag + 1]
        best = int(np.argmax(seg))
        feats.extend([(lo_lag + best) / fs, float(seg[best])])
    else:
        feats.extend([0.0, 0.0])
    return feats


FEATURE_NAMES_PER_CHANNEL = (
    "mean std min max p10 p50 p90 mean_abs_diff zcr "
    "energy band_0_1 band_1_3 band_3_5 band_5_10 dom_freq cadence_s cadence_strength"
).split()


def feature_names(channels=config.AXES) -> list[str]:
    names = []
    for ch in (*channels, "mag"):
        names += [f"{ch}_{n}" for n in FEATURE_NAMES_PER_CHANNEL]
    names += ["corr_xy", "corr_xz", "corr_yz"]
    return names


def extract_window_features(window: np.ndarray, fs: float = config.NOMINAL_SAMPLE_RATE_HZ) -> np.ndarray:
    """window: (n_samples, 3) -> 1D feature vector, ordered as feature_names()."""
    window = np.asarray(window, dtype=float)
    feats: list[float] = []
    for c in range(window.shape[1]):
        feats += _channel_features(window[:, c], fs)
    feats += _channel_features(np.linalg.norm(window, axis=1), fs)

    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(window.T)
    corr = np.nan_to_num(corr)  # a flatlined axis has undefined correlation -> 0
    feats += [corr[0, 1], corr[0, 2], corr[1, 2]]
    return np.nan_to_num(np.asarray(feats, dtype=float))


def extract_features(windows: np.ndarray, fs: float = config.NOMINAL_SAMPLE_RATE_HZ) -> np.ndarray:
    """windows: (N, n_samples, 3) -> (N, n_features)"""
    return np.vstack([extract_window_features(w, fs) for w in windows])
