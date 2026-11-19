"""
Window-level data-quality validator: run before the model, reject bad windows.

Two layers:

1. Rule-based checks, each targeting a specific physical failure and
   calibrated against what clean WISDM watch data actually looks like
   (measured on 4,000 clean windows, see README):

   flatline   longest run of consecutive identical samples (all axes).
              Clean data: never even 2 identical consecutive samples.
              Flag at >= 5 samples (0.25 s) - a frozen/stuck sensor.
   gap        longest run of perfectly linear samples (zero second
              difference on all axes). That is the signature a dropout
              leaves after resampling interpolates across it. Clean data:
              max 3. Flag at >= 10 samples (0.5 s). If raw timestamps are
              supplied, the largest inter-sample interval is checked
              directly as well.
   clipping   fraction of samples pinned at the window's per-axis extreme
              value in runs of >= 2. Clean data already saturates at about
              +/-19.6 m/s^2 (~2g) in 22% of windows (mostly jogging,
              dribbling, catch, kicking) - that is in-distribution and the
              model is trained on it - with a clean maximum of 0.08. Flag
              above 0.10: clipping worse than the sensor itself produces.
   noise      excess high-frequency noise for how much the wearer is moving.
              A robust white-noise estimate (MAD of second differences) is
              compared with a threshold that depends on low-frequency motion
              amplitude: jogging legitimately has lots of high-frequency
              energy, so a single global threshold either misses sigma <= 2
              m/s^2 noise or rejects jogging. Thresholds are the 99.5th
              percentile of the noise estimate within 10 motion-amplitude
              bins of clean training data. Added after v1 of this validator
              was measured to be blind to noise that costs the model 10-40
              F1 points; it only partially closes that gap (see README).
   magnitude  mean |a| over the window outside a band fitted on training
              data (gravity alone is 9.81 m/s^2, so a large deviation means
              a scaling/saturation fault rather than a motion).

2. A learned anomaly score: an IsolationForest on the hand-crafted features,
   fitted on clean training windows. Its raw score is converted to an
   empirical tail probability against a *held-out* calibration set of clean
   training windows from different subjects, so

       quality_score ~ Uniform(0, 1) on clean data from unseen subjects,

   and a window gets score q when only a fraction q of clean windows look
   at least as anomalous. Flag when q < alpha (default 0.01, i.e. about 1%
   false rejections on clean data by construction).

Final quality_score = anomaly tail probability if every rule passes, else 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import IsolationForest

from . import config
from .features import extract_features

FLATLINE_MAX_RUN = 5
GAP_MAX_LINEAR_RUN = 10
CLIP_MAX_FRACTION = 0.10
MAGNITUDE_MARGIN = 0.10  # widen the fitted magnitude band by 10% on each side
DEFAULT_ALPHA = 0.01
NOISE_BINS = 10
NOISE_PERCENTILE = 99.5


def _longest_true_run(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    # Run lengths via the positions where the mask changes value.
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return int((edges[1::2] - edges[0::2]).max())


def flatline_run(window: np.ndarray) -> int:
    return _longest_true_run((np.abs(np.diff(window, axis=0)) < 1e-9).all(axis=1))


def linear_run(window: np.ndarray) -> int:
    return _longest_true_run((np.abs(np.diff(window, n=2, axis=0)) < 1e-6).all(axis=1))


def clip_fraction(window: np.ndarray) -> float:
    pinned = 0
    for c in range(window.shape[1]):
        a = window[:, c]
        for extreme in (a.max(), a.min()):
            at = np.abs(a - extreme) < 1e-6
            neighbour = np.zeros_like(at)
            neighbour[1:] |= at[:-1]
            neighbour[:-1] |= at[1:]
            pinned += int((at & neighbour).sum())
    return pinned / window.size


def noise_floor_stats(window: np.ndarray) -> tuple[float, float]:
    """
    (white-noise sigma estimate, low-frequency motion amplitude), both in m/s^2.

    Noise: for white noise of std sigma, the second difference has std
    sigma*sqrt(6); the MAD makes the estimate robust to real motion peaks.
    Taking the minimum over axes keeps one high-motion axis from dominating.
    Motion: std of a 5-sample moving average (passes roughly < 2 Hz at 20 Hz).
    """
    d2 = np.diff(window, n=2, axis=0)
    mad = np.median(np.abs(d2 - np.median(d2, axis=0)), axis=0) / 0.6745
    noise = float((mad / np.sqrt(6)).min())
    kernel = np.ones(5) / 5
    low = np.column_stack([np.convolve(window[:, c] - window[:, c].mean(), kernel, mode="valid")
                           for c in range(window.shape[1])])
    return noise, float(low.std(axis=0).max())


@dataclass
class QualityReport:
    quality_score: float
    flagged: bool
    reasons: list[str] = field(default_factory=list)
    checks: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"quality_score": self.quality_score, "flagged": self.flagged,
                "reasons": self.reasons, "checks": self.checks}


class WindowValidator:
    def __init__(self, alpha: float = DEFAULT_ALPHA, seed: int = config.RANDOM_SEED,
                 sample_rate_hz: float = config.NOMINAL_SAMPLE_RATE_HZ):
        self.alpha = alpha
        self.seed = seed
        self.sample_rate_hz = sample_rate_hz

    # ---- fitting -------------------------------------------------------------
    def fit(self, windows: np.ndarray, subject_ids: np.ndarray | None = None,
            calibration_fraction: float = 0.2) -> "WindowValidator":
        """
        Fit on clean training windows. If subject_ids are given, a fraction of
        *subjects* is held out for calibration, so the score's calibration
        reflects unseen people, not just unseen windows.
        """
        rng = np.random.default_rng(self.seed)
        if subject_ids is not None:
            subs = np.unique(subject_ids)
            n_cal = max(1, int(round(calibration_fraction * len(subs))))
            cal_subs = rng.choice(subs, size=n_cal, replace=False)
            cal_mask = np.isin(subject_ids, cal_subs)
        else:
            cal_mask = rng.random(len(windows)) < calibration_fraction

        feats = extract_features(windows)
        self.forest_ = IsolationForest(n_estimators=200, max_samples=min(512, int((~cal_mask).sum())), random_state=self.seed, n_jobs=-1)
        self.forest_.fit(feats[~cal_mask])
        self.calibration_scores_ = np.sort(self.forest_.score_samples(feats[cal_mask]))

        mag = np.linalg.norm(windows, axis=2).mean(axis=1)
        lo, hi = np.percentile(mag, [0.1, 99.9])
        self.magnitude_band_ = (float(lo * (1 - MAGNITUDE_MARGIN)), float(hi * (1 + MAGNITUDE_MARGIN)))

        stats = np.array([noise_floor_stats(w) for w in windows])
        log_noise, log_motion = np.log(stats[:, 0] + 1e-6), np.log(stats[:, 1] + 1e-6)
        edges = np.quantile(log_motion, np.linspace(0, 1, NOISE_BINS + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        self.noise_bin_edges_ = edges
        self.noise_thresholds_ = np.array([
            np.percentile(log_noise[(log_motion >= edges[i]) & (log_motion < edges[i + 1])], NOISE_PERCENTILE)
            for i in range(NOISE_BINS)])
        return self

    def _noise_excess(self, window: np.ndarray) -> tuple[float, bool]:
        noise, motion = noise_floor_stats(window)
        b = int(np.clip(np.searchsorted(self.noise_bin_edges_, np.log(motion + 1e-6), side="right") - 1,
                        0, NOISE_BINS - 1))
        return noise, bool(np.log(noise + 1e-6) > self.noise_thresholds_[b])

    # ---- scoring -------------------------------------------------------------
    def anomaly_tail_probability(self, feats: np.ndarray) -> np.ndarray:
        """Fraction of clean calibration windows scoring as anomalous or more (lower = more anomalous)."""
        raw = self.forest_.score_samples(feats)
        ranks = np.searchsorted(self.calibration_scores_, raw, side="right")
        return (ranks + 1) / (len(self.calibration_scores_) + 1)

    def rule_checks(self, window: np.ndarray, timestamps_ns: np.ndarray | None = None) -> tuple[dict, list[str]]:
        checks = {
            "flatline_run": flatline_run(window),
            "linear_run": linear_run(window),
            "clip_fraction": round(clip_fraction(window), 4),
            "mean_magnitude": round(float(np.linalg.norm(window, axis=1).mean()), 3),
        }
        reasons = []
        if checks["flatline_run"] >= FLATLINE_MAX_RUN:
            reasons.append("flatline")
        if checks["linear_run"] >= GAP_MAX_LINEAR_RUN:
            reasons.append("gap")
        if timestamps_ns is not None and len(timestamps_ns) > 1:
            max_gap_ms = float(np.diff(np.asarray(timestamps_ns, dtype=np.int64)).max() / 1e6)
            checks["max_timestamp_gap_ms"] = round(max_gap_ms, 2)
            if max_gap_ms > config.MAX_GAP_MS:
                reasons.append("timestamp_gap")
        if checks["clip_fraction"] > CLIP_MAX_FRACTION:
            reasons.append("clipping")
        noise_sigma, noisy = self._noise_excess(window)
        checks["noise_sigma_estimate"] = round(noise_sigma, 4)
        if noisy:
            reasons.append("noise")
        lo, hi = self.magnitude_band_
        if not lo <= checks["mean_magnitude"] <= hi:
            reasons.append("magnitude_out_of_range")
        return checks, reasons

    def validate(self, window: np.ndarray, timestamps_ns: np.ndarray | None = None) -> QualityReport:
        return self.validate_batch(window[None], None if timestamps_ns is None else [timestamps_ns])[0]

    def validate_batch(self, windows: np.ndarray, timestamps: list | None = None,
                       alpha: float | None = None, features: np.ndarray | None = None) -> list[QualityReport]:
        alpha = self.alpha if alpha is None else alpha
        windows = np.asarray(windows, dtype=float)
        feats = extract_features(windows) if features is None else features
        tail = self.anomaly_tail_probability(feats)
        reports = []
        for i, w in enumerate(windows):
            checks, reasons = self.rule_checks(w, None if timestamps is None else timestamps[i])
            checks["anomaly_tail_probability"] = round(float(tail[i]), 4)
            if tail[i] < alpha:
                reasons.append("anomalous_pattern")
            score = 0.0 if any(r != "anomalous_pattern" for r in reasons) else float(tail[i])
            reports.append(QualityReport(round(score, 4), bool(reasons), reasons, checks))
        return reports

    def scores(self, windows: np.ndarray) -> np.ndarray:
        """Vector of quality scores, for threshold sweeps."""
        return np.array([r.quality_score for r in self.validate_batch(windows)])
