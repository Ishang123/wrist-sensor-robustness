"""
Load, clean, resample, and window the WISDM smartwatch data.

Raw file format (one .txt per subject per sensor, e.g.
``data_1600_accel_watch.txt``), each line::

    subject_id,activity_code,timestamp_ns,x,y,z;

Activities appear as contiguous blocks (~3 minutes each) within a file, not
interleaved, and each block's timestamps are monotonically increasing - but
the file is not globally time-sorted across blocks, so sorting must happen
per (subject, activity).

Pipeline: load raw rows -> basic cleaning -> resample to a fixed rate ->
window into fixed-length, possibly-overlapping segments. Each step is a
small, independently testable function.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
import pandas as pd

from . import config

RAW_COLUMNS = ["subject_id", "activity", "timestamp", "x", "y", "z"]
Sensor = Literal["accel", "gyro"]


def load_activity_key(path: Path = config.ACTIVITY_KEY_PATH) -> dict[str, str]:
    """Parse activity_key.txt ("walking = A") into {code: name}, e.g. {"A": "walking"}."""
    mapping: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        name, code = line.split("=", 1)
        mapping[code.strip()] = name.strip()
    return mapping


def _sensor_dir(sensor: Sensor) -> Path:
    return config.RAW_WATCH_ACCEL_DIR if sensor == "accel" else config.RAW_WATCH_GYRO_DIR


def available_subject_ids(sensor: Sensor = "accel") -> list[int]:
    """Subject IDs that have a raw file for this sensor, sorted ascending."""
    ids = []
    for f in _sensor_dir(sensor).glob(f"data_*_{sensor}_watch.txt"):
        ids.append(int(f.stem.split("_")[1]))
    return sorted(ids)


def load_subject_sensor(subject_id: int, sensor: Sensor = "accel") -> pd.DataFrame:
    """
    Load one subject's raw file for one sensor.

    Returns columns [subject_id, activity, timestamp, x, y, z], timestamp in
    nanoseconds as in the source file. The trailing ';' on the last numeric
    field is stripped. Rows are NOT sorted here (callers that need temporal
    order should sort per-activity, since blocks aren't globally ordered).
    """
    path = _sensor_dir(sensor) / f"data_{subject_id}_{sensor}_watch.txt"
    df = pd.read_csv(path, header=None, names=RAW_COLUMNS, engine="python", on_bad_lines="skip")
    # The 'z' column carries the trailing semicolon from the source format.
    df["z"] = df["z"].astype(str).str.rstrip(";").astype(float)
    for col in ("x", "y"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("Int64")
    return df


def clean_sensor_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning shared by every sensor load:
      - drop rows with NaN/invalid numeric fields (malformed source lines)
      - drop exact duplicate (subject_id, activity, timestamp) rows
      - sort within each (subject_id, activity) block by timestamp
    """
    before = len(df)
    df = df.dropna(subset=["timestamp", "x", "y", "z"]).copy()
    df["timestamp"] = df["timestamp"].astype(np.int64)
    df = df.drop_duplicates(subset=["subject_id", "activity", "timestamp"])
    # Sorting by the group keys first, then timestamp, sorts within each
    # (subject_id, activity) block without needing a groupby.apply.
    df = df.sort_values(["subject_id", "activity", "timestamp"]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        df.attrs["n_dropped_rows"] = df.attrs.get("n_dropped_rows", 0) + dropped
    return df


def merge_accel_gyro(accel: pd.DataFrame, gyro: pd.DataFrame, tolerance_ms: float = 25.0) -> pd.DataFrame:
    """
    Align accelerometer and gyroscope streams by nearest timestamp within
    each (subject_id, activity) block.

    The two sensors are sampled independently (confirmed on this dataset: same
    ~20 Hz nominal rate, but offset by tens of ms), so there is no shared
    timestamp to join on exactly - this does an as-of (nearest, bounded)
    merge instead. Rows with no gyro sample within `tolerance_ms` are dropped,
    since fabricating a gyro reading would misrepresent sensor data as
    measured when it wasn't.

    Output columns: subject_id, activity, timestamp, x, y, z, gx, gy, gz.
    """
    merged_blocks = []
    gyro_renamed = gyro.rename(columns={"x": "gx", "y": "gy", "z": "gz"})

    for (sid, act), a_block in accel.groupby(["subject_id", "activity"], sort=False):
        g_block = gyro_renamed[
            (gyro_renamed["subject_id"] == sid) & (gyro_renamed["activity"] == act)
        ]
        if g_block.empty:
            continue
        merged = pd.merge_asof(
            a_block.sort_values("timestamp"),
            g_block.sort_values("timestamp")[["timestamp", "gx", "gy", "gz"]],
            on="timestamp",
            direction="nearest",
            tolerance=int(tolerance_ms * 1e6),  # ms -> ns
        )
        merged_blocks.append(merged.dropna(subset=["gx", "gy", "gz"]))

    if not merged_blocks:
        raise ValueError("No overlapping (subject, activity) blocks between accel and gyro.")
    return pd.concat(merged_blocks, ignore_index=True)


def split_at_gaps(df: pd.DataFrame, max_gap_ms: float = config.MAX_GAP_MS) -> pd.DataFrame:
    """
    Add a `segment` column: a new segment starts wherever consecutive samples
    in a (subject_id, activity) block are more than `max_gap_ms` apart.

    Measured on this dataset: the median inter-sample interval is 49.9 ms, but
    there are 460 gaps over 100 ms, 5 over 1 s, and one of 56.7 s. Resampling
    straight across those would fill them with fabricated linear ramps, so
    each contiguous segment is resampled and windowed on its own and no
    window ever spans a real gap.
    """
    df = df.sort_values(["subject_id", "activity", "timestamp"]).reset_index(drop=True)
    new_block = (df["subject_id"] != df["subject_id"].shift()) | (df["activity"] != df["activity"].shift())
    big_gap = df["timestamp"].diff() > max_gap_ms * 1e6
    df["segment"] = (new_block | big_gap).cumsum() - 1
    return df


def resample_to_fixed_rate(
    df: pd.DataFrame,
    target_hz: int = config.NOMINAL_SAMPLE_RATE_HZ,
    max_gap_ms: float = config.MAX_GAP_MS,
) -> pd.DataFrame:
    """
    Resample each contiguous segment to a fixed sample rate by linear
    interpolation onto an evenly spaced time grid.

    Segments come from split_at_gaps, so interpolation only ever bridges the
    small per-sample jitter of the raw ~20 Hz stream (and gaps up to
    max_gap_ms), never a real recording gap. Interpolating onto a uniform
    grid is what makes fixed-length windows at target_hz well defined.
    """
    if "segment" not in df.columns:
        df = split_at_gaps(df, max_gap_ms=max_gap_ms)
    value_cols = [c for c in df.columns if c not in ("subject_id", "activity", "timestamp", "segment")]
    period_ns = int(1e9 / target_hz)

    out_blocks = []
    for (sid, act, seg), block in df.groupby(["subject_id", "activity", "segment"], sort=False):
        t0, t1 = int(block["timestamp"].iloc[0]), int(block["timestamp"].iloc[-1])
        if t1 <= t0 or len(block) < 2:
            continue
        grid = np.arange(t0, t1, period_ns, dtype=np.int64)
        resampled = {"subject_id": sid, "activity": act, "segment": seg, "timestamp": grid}
        for col in value_cols:
            resampled[col] = np.interp(grid, block["timestamp"].to_numpy(), block[col].to_numpy())
        out_blocks.append(pd.DataFrame(resampled))

    return pd.concat(out_blocks, ignore_index=True)


@dataclass
class Window:
    """One fixed-length windowed segment, with its provenance intact for grouping/auditing."""

    subject_id: int
    activity: str
    start_timestamp: int
    data: np.ndarray  # shape (n_samples, n_channels), channel order given by `channels`
    channels: tuple[str, ...]


def window_signal(
    df: pd.DataFrame,
    window_samples: int = config.WINDOW_SAMPLES,
    overlap: float = config.WINDOW_OVERLAP,
    channels: tuple[str, ...] = config.AXES,
) -> Iterator[Window]:
    """
    Slice a resampled (subject_id, activity) DataFrame into fixed-length,
    overlapping windows.

    Windows never cross a segment boundary (see split_at_gaps).
    A window is only emitted if it is fully observed (no padding ever
    introduced by this function) - a trailing partial segment shorter than
    `window_samples` is dropped, never zero-padded, so every window the rest
    of the pipeline sees genuinely contains window_samples real samples.
    """
    step = max(1, int(round(window_samples * (1 - overlap))))

    keys = ["subject_id", "activity"] + (["segment"] if "segment" in df.columns else [])
    for key, block in df.groupby(keys, sort=False):
        sid, act = key[0], key[1]
        block = block.sort_values("timestamp")
        arr = block[list(channels)].to_numpy()
        ts = block["timestamp"].to_numpy()
        n = len(block)
        for start in range(0, n - window_samples + 1, step):
            end = start + window_samples
            yield Window(
                subject_id=int(sid),
                activity=str(act),
                start_timestamp=int(ts[start]),
                data=arr[start:end].copy(),
                channels=channels,
            )


def windows_to_frame(windows: list[Window]) -> pd.DataFrame:
    """Pack a list of Windows into one DataFrame: metadata columns + a 3D array column."""
    return pd.DataFrame(
        {
            "subject_id": [w.subject_id for w in windows],
            "activity": [w.activity for w in windows],
            "start_timestamp": [w.start_timestamp for w in windows],
            "data": [w.data for w in windows],
        }
    )


def build_dataset(
    subject_ids: list[int] | None = None,
    sensor: Sensor = "accel",
    target_hz: int = config.NOMINAL_SAMPLE_RATE_HZ,
    window_samples: int = config.WINDOW_SAMPLES,
    overlap: float = config.WINDOW_OVERLAP,
) -> pd.DataFrame:
    """
    End-to-end: load -> clean -> resample -> window, for a set of subjects.

    Returns one row per window: subject_id, activity, start_timestamp, data
    (np.ndarray of shape (window_samples, 3)).
    """
    if subject_ids is None:
        subject_ids = available_subject_ids(sensor)

    all_windows: list[Window] = []
    for sid in subject_ids:
        raw = load_subject_sensor(sid, sensor=sensor)
        clean = clean_sensor_frame(raw)
        resampled = resample_to_fixed_rate(clean, target_hz=target_hz)
        all_windows.extend(
            window_signal(resampled, window_samples=window_samples, overlap=overlap)
        )
    return windows_to_frame(all_windows)


WINDOW_CACHE_PATH = config.DATA_DIR / "windows_accel.npz"


def save_windows_cache(df: pd.DataFrame, path: Path = WINDOW_CACHE_PATH) -> None:
    np.savez_compressed(
        path,
        X=np.stack(df["data"].to_numpy()),
        subject_id=df["subject_id"].to_numpy(),
        activity=df["activity"].to_numpy().astype(str),
        start_ts=df["start_timestamp"].to_numpy(),
    )


def load_windows_cache(path: Path = WINDOW_CACHE_PATH, rebuild: bool = False):
    """Return (X, activity_labels, subject_ids), building the cache from raw files if needed."""
    if rebuild or not path.exists():
        save_windows_cache(build_dataset(), path)
    z = np.load(path, allow_pickle=False)
    return z["X"], z["activity"].astype(str), z["subject_id"]
