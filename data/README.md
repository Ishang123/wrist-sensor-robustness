# data/

Not committed (see `.gitignore`). To reproduce:

1. Download `wisdm-dataset.zip` from https://archive.ics.uci.edu/dataset/507 (CC BY 4.0; cite Weiss 2019, DOI 10.24432/C5HK59).
2. Copy `wisdm-dataset/raw/watch/accel/` to `data/raw/watch/accel/` (51 files, `data_<subject>_accel_watch.txt`).
3. `activity_key.txt` (activity code -> name) is already here.
4. `make data` builds `data/windows_accel.npz`, the 31,014-window cache everything else reads.
