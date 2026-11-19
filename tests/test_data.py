import numpy as np
import pandas as pd
import pytest

from conftest import requires_real_data
from wrist_sensor_robustness import config, data as d


def _write_raw(tmp_path, monkeypatch, lines):
    accel = tmp_path / "accel"
    accel.mkdir()
    (accel / "data_1600_accel_watch.txt").write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(config, "RAW_WATCH_ACCEL_DIR", accel)


def test_load_parses_trailing_semicolon_and_skips_malformed(tmp_path, monkeypatch):
    _write_raw(tmp_path, monkeypatch, [
        "1600,A,1000000000,1.0,2.0,3.5;",
        "1600,A,1050000000,1.1,2.1,3.6;",
        "this,is,not,a,valid;",
        "1600,A,1100000000,1.2,2.2,3.7;",
    ])
    df = d.clean_sensor_frame(d.load_subject_sensor(1600))
    assert list(df.columns[:6]) == d.RAW_COLUMNS
    assert len(df) == 3
    assert df["z"].tolist() == [3.5, 3.6, 3.7]
    assert df["timestamp"].is_monotonic_increasing


def test_clean_drops_duplicates_and_sorts_within_activity():
    df = pd.DataFrame({"subject_id": [1, 1, 1, 1], "activity": ["A", "A", "A", "B"],
                       "timestamp": [300, 100, 100, 50], "x": [3., 1., 1., 9.], "y": 0., "z": 0.})
    out = d.clean_sensor_frame(df)
    a = out[out.activity == "A"]
    assert a["timestamp"].tolist() == [100, 300]


def test_split_at_gaps_starts_new_segment_after_large_gap():
    ts = np.array([0, 50, 100, 2_000, 2_050]) * 1_000_000  # ms -> ns; 1.9 s gap after the 3rd sample
    df = pd.DataFrame({"subject_id": 1, "activity": "A", "timestamp": ts, "x": 0., "y": 0., "z": 0.})
    seg = d.split_at_gaps(df, max_gap_ms=250)["segment"].tolist()
    assert seg[:3] == [seg[0]] * 3 and seg[3] == seg[4] != seg[0]


def test_resample_produces_uniform_grid_and_never_bridges_gaps():
    rng = np.random.default_rng(0)
    # 4 s of jittery ~20 Hz data, a 3 s gap, then 4 s more
    t1 = np.cumsum(rng.uniform(48, 52, 80)) * 1e6
    t2 = t1[-1] + 3e9 + np.cumsum(rng.uniform(48, 52, 80)) * 1e6
    ts = np.concatenate([t1, t2]).astype(np.int64)
    df = pd.DataFrame({"subject_id": 1, "activity": "A", "timestamp": ts,
                       "x": np.arange(160.), "y": 0., "z": 0.})
    out = d.resample_to_fixed_rate(df, target_hz=20, max_gap_ms=250)
    for _, seg in out.groupby("segment"):
        assert np.all(np.diff(seg["timestamp"]) == 50_000_000)
    assert out["segment"].nunique() == 2


def test_window_count_overlap_and_no_padding():
    n = 1000
    df = pd.DataFrame({"subject_id": 1, "activity": "A", "segment": 0,
                       "timestamp": np.arange(n) * 50_000_000, "x": np.arange(n, dtype=float), "y": 0., "z": 0.})
    wins = list(d.window_signal(df, window_samples=200, overlap=0.5))
    assert len(wins) == (n - 200) // 100 + 1
    assert all(w.data.shape == (200, 3) for w in wins)
    # 50% overlap: second window starts 100 samples after the first
    assert wins[1].data[0, 0] - wins[0].data[0, 0] == 100


def test_windows_never_cross_segments():
    df = pd.DataFrame({"subject_id": 1, "activity": "A",
                       "segment": [0] * 250 + [1] * 250,
                       "timestamp": np.arange(500) * 50_000_000,
                       "x": [0.] * 250 + [1.] * 250, "y": 0., "z": 0.})
    for w in d.window_signal(df, window_samples=200, overlap=0.5):
        assert len(np.unique(w.data[:, 0])) == 1


def test_activity_key_parses(tmp_path):
    p = tmp_path / "key.txt"
    p.write_text("walking = A\njogging = B\n\n")
    assert d.load_activity_key(p) == {"A": "walking", "B": "jogging"}


@requires_real_data
def test_real_subject_has_18_activities_at_about_20hz():
    df = d.clean_sensor_frame(d.load_subject_sensor(1600))
    assert df["activity"].nunique() == 18
    dt_ms = np.diff(df[df.activity == "A"]["timestamp"].to_numpy()) / 1e6
    assert 49 < np.median(dt_ms) < 51


@requires_real_data
def test_real_subject_ids_match_documentation():
    assert d.available_subject_ids() == list(range(1600, 1651))
