| model | gate | windows rejected % | accuracy % | macro-F1 % | corrupted share of kept % |
|---|---|---|---|---|---|
| GBM (features) | none | 0.0 | 69.2 ± 5.3 | 69.6 ± 4.9 | 30.0 (by construction) |
| GBM (features) | validator | 9.9 ± 1.0 | 71.2 ± 5.5 | 71.8 ± 5.0 | 22.4 |
| GBM (features) | confidence | 9.9 ± 1.0 | 74.3 ± 5.6 | 73.8 ± 5.3 | 27.3 |
| GBM (features) | hybrid: rules + confidence | 9.9 ± 1.0 | 72.0 ± 5.5 | 72.1 ± 5.1 | 23.0 |
| 1D CNN | none | 0.0 | 61.1 ± 5.4 | 61.3 ± 5.2 | 30.0 (by construction) |
| 1D CNN | validator | 9.9 ± 1.0 | 62.6 ± 5.6 | 63.1 ± 5.4 | 22.4 |
| 1D CNN | confidence | 9.9 ± 1.0 | 65.2 ± 5.6 | 64.3 ± 5.4 | 28.6 |
| 1D CNN | hybrid: rules + confidence | 9.9 ± 1.0 | 63.1 ± 5.8 | 63.1 ± 5.5 | 23.6 |
