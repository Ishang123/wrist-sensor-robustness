# End-to-end: make data && make baseline && make robustness && make analyze
PY ?= python3

.PHONY: install data baseline robustness analyze train-final serve test all clean

install:
	$(PY) -m pip install -r requirements.txt

data:          ## raw WISDM watch files -> cached 10 s windows (data/windows_accel.npz)
	$(PY) -c "import sys; sys.path.insert(0,'src'); from wrist_sensor_robustness.data import load_windows_cache; X,_,_=load_windows_cache(rebuild=True); print(X.shape)"

baseline:      ## subject-wise + random-window 5-fold CV for both models (~45 min on 2 cores)
	$(PY) scripts/run_baseline.py

robustness:    ## corruption sweep + validator records, reuses baseline fold models
	$(PY) scripts/run_robustness.py

analyze:       ## tables and plots from saved results
	$(PY) scripts/analyze_results.py

train-final:   ## artifacts the API serves (fit on all subjects)
	$(PY) scripts/train_final.py

serve: train-final
	PYTHONPATH=src $(PY) -m uvicorn wrist_sensor_robustness.api:app --port 8000

test:
	$(PY) -m pytest -q

all: data baseline robustness analyze train-final test

clean:
	rm -rf results/models results/robustness_records.csv.gz results/logs
