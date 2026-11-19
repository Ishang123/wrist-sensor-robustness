"""Project-wide constants. One place to change the dataset path, window size, etc."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_WATCH_ACCEL_DIR = DATA_DIR / "raw" / "watch" / "accel"
RAW_WATCH_GYRO_DIR = DATA_DIR / "raw" / "watch" / "gyro"
ACTIVITY_KEY_PATH = DATA_DIR / "activity_key.txt"
RESULTS_DIR = PROJECT_ROOT / "results"

# WISDM smartwatch subject IDs, per the dataset documentation.
SUBJECT_ID_MIN = 1600
SUBJECT_ID_MAX = 1650

# Nominal sampling rate of the WISDM smartwatch stream (Hz). Actual per-subject
# rate is measured from timestamps, not assumed - see data.resample_to_fixed_rate.
NOMINAL_SAMPLE_RATE_HZ = 20

# Windowing.
WINDOW_SECONDS = 10
WINDOW_OVERLAP = 0.5  # fraction of a window that overlaps with the next one
WINDOW_SAMPLES = WINDOW_SECONDS * NOMINAL_SAMPLE_RATE_HZ  # 200 at 20 Hz

# Split a recording wherever consecutive raw samples are further apart than this
# (5 nominal sample periods), instead of interpolating across the gap.
MAX_GAP_MS = 250

RANDOM_SEED = 42

AXES = ("x", "y", "z")
