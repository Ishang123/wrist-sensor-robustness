| model | split | accuracy % | macro-F1 % |
|---|---|---|---|
| GBM (features) | subject-wise (5-fold, by person) | 75.2 ± 4.7 | 75.7 ± 4.2 |
| GBM (features) | random windows (leaky) | 93.6 ± 0.3 | 93.6 ± 0.2 |
| GBM (features) | **leakage inflation** | +18.3 pts | +17.8 pts |
| 1D CNN | subject-wise (5-fold, by person) | 67.3 ± 5.4 | 67.6 ± 5.1 |
| 1D CNN | random windows (leaky) | 83.2 ± 0.7 | 83.3 ± 0.6 |
| 1D CNN | **leakage inflation** | +15.9 pts | +15.7 pts |
