# 1. Evaluation

## 1H

| Method | NMRShiftDB2 (L_atom) MAE | NMRShiftDB2 (L_atom) RMSE | NMRShiftDB2 (L_mol) MAE | NMRShiftDB2 (L_mol) RMSE | ShiftDB-Lit (L_mol) MAE | ShiftDB-Lit (L_mol) RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HOSE | 0.33 | — | — | — | — | — |
| GCN | 0.28 | — | — | — | — | — |
| FCG | 0.224 | — | — | — | — | — |
| SGNN | 0.216 | 0.484 | — | — | — | — |
| GT-NMR | (0.158)* | (0.293)* | — | — | — | — |
| NMRNet (Baseline) | *0.1972* | *0.4564* | *0.1761* | *0.3896* | *0.1395* | *0.2790* |
| NMRNet (Semi-supervised) | **0.1709** | **0.4337** | **0.1492** | **0.3620** | **0.0559** | **0.1846** |
| *Δ vs baseline* | (↓ 13.4%) | (↓ 5.0%) | (↓ 15.3%) | (↓ 7.1%) | (↓ 59.9%) | (↓ 33.8%) |

\*GT-NMR predicts only hydrogens bonded to carbon in the original work, which is not directly comparable to the full evaluation.

## 13C

| Method | NMRShiftDB2 (L_atom) MAE | NMRShiftDB2 (L_atom) RMSE | NMRShiftDB2 (L_mol) MAE | NMRShiftDB2 (L_mol) RMSE | ShiftDB-Lit (L_mol) MAE | ShiftDB-Lit (L_mol) RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HOSE | 2.85 | — | — | — | — | — |
| GCN | 1.43 | — | — | — | — | — |
| FCG | 1.355 | — | — | — | — | — |
| SGNN | 1.271 | 2.232 | — | — | — | — |
| GT-NMR | 1.189 | 2.206 | — | — | — | — |
| NMRNet (Baseline) | *1.1518* | *2.1398* | *1.0143* | *1.8513* | *1.2591* | *2.9207* |
| NMRNet (Semi-supervised) | **0.9270** | **1.9128** | **0.7765** | **1.5629** | **0.5060** | **2.3494** |
| *Δ vs baseline* | (↓ 19.6%) | (↓ 10.5%) | (↓ 23.4%) | (↓ 15.6%) | (↓ 59.8%) | (↓ 19.6%) |


# 2. More Solvents

## 1H

| Solvent | Num. molecules | With incorporation MAE | With incorporation RMSE | Without incorporation MAE | Without incorporation RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| CDCl3 | 162,509 | **0.0475 (↓ 5.4%)** | **0.1580 (↓ 5.3%)** | 0.0502 | 0.1668 |
| DMSO-d6 | 11,623 | **0.0658 (↓ 46.8%)** | **0.2185 (↓ 36.0%)** | 0.1237 | 0.3415 |
| Others | 5,553 | **0.0996 (↓ 9.5%)** | **0.2198 (↓ 25.2%)** | 0.1100 | 0.2938 |
| All | 176,985 | **0.0501 (↓ 10.8%)** | **0.1665 (↓ 10.5%)** | 0.0562 | 0.1861 |

## 13C

| Solvent | Num. molecules | With incorporation MAE | With incorporation RMSE | Without incorporation MAE | Without incorporation RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| CDCl3 | 126,364 | **0.4775 (↓ 2.6%)** | **2.2990 (↓ 0.3%)** | 0.4903 | 2.3056 |
| DMSO-d6 | 9,026 | **0.6755 (↓ 17.8%)** | **2.5298 (↓ 2.0%)** | 0.8223 | 2.5818 |
| Others | 5,485 | **0.8684 (↓ 9.0%)** | **3.0113 (↓ 1.4%)** | 0.9547 | 3.0530 |
| All | 140,875 | **0.5042 (↓ 4.5%)** | **2.3440 (↓ 0.2%)** | 0.5281 | 2.3494 |


# 3. Training Strategy

Ablation study on the effect of the training dataset. **DB2** refers to NMRShiftDB2; **DB-Lit** refers to ShiftDB-Lit. A `--` entry means the corresponding dataset or loss is not used for that training path.

## 1H

| No. | Supervised | Weakly-supervised | NMRShiftDB2 (L_atom) MAE | NMRShiftDB2 (L_atom) RMSE | NMRShiftDB2 (L_mol) MAE | NMRShiftDB2 (L_mol) RMSE | ShiftDB-Lit (L_mol) MAE | ShiftDB-Lit (L_mol) RMSE |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | DB2 | -- | 0.1972 | 0.4564 | 0.1761 | 0.3896 | 0.1395 | 0.2790 |
| 2 | -- | DB-Lit | 0.2412 | 0.5600 | 0.2103 | 0.4533 | 0.0543 | 0.1849 |
| 3 | -- | DB2 | 0.2308 | 0.4902 | 0.1963 | 0.4051 | 0.1439 | 0.2835 |
| 4 | DB2 | DB2 | 0.2152 | 0.4844 | 0.1829 | 0.3968 | 0.1413 | 0.2840 |
| 5 | DB2 | DB-Lit | **0.1709** | **0.4337** | **0.1492** | **0.3620** | **0.0559** | **0.1846** |

## 13C

| No. | Supervised | Weakly-supervised | NMRShiftDB2 ($L_{\text{atom}}$) MAE | NMRShiftDB2 ($L_{\text{atom}}$) RMSE | NMRShiftDB2 ($L_{\text{mol}}$) MAE | NMRShiftDB2 ($L_{\text{mol}}$) RMSE | ShiftDB-Lit ($L_{\text{mol}}$) MAE | ShiftDB-Lit ($L_{\text{mol}}$) RMSE |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | DB2 | -- | 1.1518 | 2.1398 | 1.0143 | 1.8513 | 1.2591 | 2.9207 |
| 2 | -- | DB-Lit | 1.5214 | 4.6658 | 1.2931 | 3.1443 | 0.9965 | 2.5962 |
| 3 | -- | DB2 | 2.3848 | 4.0615 | 2.1943 | 3.7542 | 2.0393 | 3.9687 |
| 4 | DB2 | DB2 | 1.1503 | 2.2251 | 0.9753 | 1.8541 | 1.1730 | 2.9139 |
| 5 | DB2 | DB-Lit | **0.9270** | **1.9128** | **0.7765** | **1.5629** | **0.5060** | **2.3494** |


# 4. ID vs OOD Evaluation

| Method | NMRShiftDB2 (L_atom) MAE | NMRShiftDB2 (L_atom) RMSE | NMRShiftDB2 (L_mol) MAE | NMRShiftDB2 (L_mol) RMSE | ShiftDB-Lit (L_mol) MAE | ShiftDB-Lit (L_mol) RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Supervised (NMRShiftDB2) | *1.1518* | *2.1398* | *1.0143* | *1.8513* | *1.2591* | *2.9207* |
| Semi-supervised (NMRShiftDB2 + ShiftDB-Lit) | **0.9270** | **1.9128** | **0.7765** | **1.5629** | **0.5060** | **2.3494** |

