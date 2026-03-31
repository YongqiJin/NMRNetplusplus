<!-- # Response to Reviewer wmHV (review3)

## Weaknesses (W1–W6) -->

Thank you for your time and thoughtful reviews! We address your comments point by point as follows.

**("W" refers to Weaknesses, "Q" refers to Key Questions)**


<!-- > **W1 (Major Soundness).** The ShiftDB-Lit test set evaluation conflates two effects. The NMRNet baseline is trained only on NMRShiftDB2, making ShiftDB-Lit out-of-distribution for it, while the semi-supervised model trains on ShiftDB-Lit, making it in-distribution. The reported reductions ¹H MAE from 0.1395 to 0.0559 (↓59.9%) and ¹³C MAE from 1.2591 to 0.5060 (↓59.8%) therefore reflect both the value of semi-supervised learning and simple distribution coverage. The authors acknowledge this (lines 283–293) but do not attempt to disentangle the two contributions for instance, by evaluating on a held-out scaffold split that is OOD for both models. This is fixable: a scaffold-based partition of ShiftDB-Lit that excludes training scaffolds from both models would isolate the semi-supervised learning gain.


> **W2 (Major Soundness).** ShiftDB-Lit uses a random 4:1 train/test split (Section 4.1) rather than a scaffold split. For a dataset of ~1.6M molecules, random splitting almost certainly places structurally similar molecules on both sides of the partition, inflating reported metrics. The NMRShiftDB2 results use a pre-defined benchmark split and are less affected, but the ShiftDB-Lit numbers which show the most dramatic gains (59.9% and 59.8% MAE reductions) and carry the paper's strongest narrative claims are the most vulnerable to this concern. This is also fixable by re-evaluating with a scaffold split. -->


#### Response-1 (W1: OOD vs ID, W2 & Q1: scaffold split)

We thank the reviewer for raising this point and for the constructive suggestion to use **scaffold-style** scrutiny of generalization. We acknowledge that a random 4:1 train/test split on ShiftDB-Lit can place **structurally similar** molecules on both sides of the partition. To better disentangle the contributions of data coverage and semi-supervised learning, we stratify ShiftDB-Lit **test** molecules by their maximum Tanimoto similarity (2048-bit Morgan fingerprints) to the union of training structures from NMRShiftDB2 and ShiftDB-Lit: **similar** (≥0.7), **intermediate** (0.5-0.7), and **dissimilar** (<0.5).

Table A2 summarizes the results. On the **similar** subset, the semi-supervised model gains the most—consistent with narrowing the OOD–ID gap when training better covers the relevant chemical space. On the **dissimilar** subset, the semi-supervised model still clearly outperforms the supervised baseline, but the improvement is less pronounced than on the similar subset.

**Table A2: OOD-vs-ID evaluation.**

- **¹H**
||dissimilar||intermediate||similar||
|-|-:|-:|-:|-:|-:|-:|
||MAE|RMSE|MAE|RMSE|MAE|RMSE|
|Baseline|0.1964|0.4265|0.1478|0.3001|0.1332|0.2659|
|Semi-supervised|0.1207|0.3620|0.0669|0.2110|0.0491|0.1656|
|↓|38.5%|15.1%|54.7%|29.7%|63.1%|37.7%|

- **¹³C**
||dissimilar||intermediate||similar||
|-|-:|-:|-:|-:|-:|-:|
||MAE|RMSE|MAE|RMSE|MAE|RMSE|
|Baseline|2.2113|5.2994|1.4188|3.4229|1.1360|2.5034|
|Semi-supervised|1.3924|4.7814|0.6451|2.8615|0.4032|1.9066|
|↓|37.0%|9.8%|54.5%|16.4%|64.5%|23.8%|

We will add these detailed results in the revised version.


#### Response-2 (W2: scaffold split)
See Response-1.


<!-- > **W3 (Major Soundness).** The solvent embedding is limited to three categories: CDCl₃ (89.1% of data, 162,509 test molecules), DMSO-d₆ (5.7%, 11,623 test molecules), and "others" (5,553 test molecules for ¹H). Table 3 shows the catch-all "others" category retains the highest MAE (0.0996 for ¹H, 0.8684 for ¹³C with solvent incorporation), and its improvement over the solvent-agnostic baseline is modest (↓9.5% for ¹H, ↓9.0% for ¹³C) compared to the gains for DMSO-d₆ (↓46.8% for ¹H, ↓17.8% for ¹³C). There is no analysis of (a) whether the "others" embedding does meaningful work beyond what the model learns without it, or (b) whether finer solvent granularity (e.g., 5–10 categories) would improve performance. Given that ShiftDB-Lit presumably contains solvent identity metadata beyond these three bins, the coarse grouping appears to be an underexplored design choice rather than a hard constraint. A targeted ablation comparing 3-bin vs. finer-grained solvent embeddings would address this. -->


#### Response-3 (W3 & Q2: fine-grained solvent embedding)

Thank the reviewer for raising this point. The original catagorization is not very fine-grained, and we have added more detailed results in Table A3.

Table A3: detailed results of comparison between with and without solvent embedding.

| Solvent | ¹H Num | ¹H w/ solv. embed. | ¹H w/o solv. embed. | ¹³C Num | ¹³C w/ solv. embed. | ¹³C w/o solv. embed. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CDCl3 | 162,509 | **0.0474 (↓ 5.4%)** | 0.0502 | 126,364 | **0.4772 (↓ 2.7%)** | 0.4903 |
| DMSO-d6| 11,623 | **0.0654 (↓ 46.8%)** | 0.1237 | 9,026 | **0.6779 (↓ 17.6%)** | 0.8223 |
| CD3COCD3 | 1,263 | **0.0659 (↓ 46.0%)** | 0.1221 | 1,123 | **0.8120 (↓ 22.3%)** | 1.0457 |
| CD2Cl2 | 1,030 | **0.0539 (↓ 9.7%)** | 0.0597 | 730 | **0.6366 (↓ 11.8%)** | 0.7215 |
| C6D6 | 589 | **0.0927 (↓ 59.6%)** | 0.2294 | 424 | **0.8924 (↓ 7.5%)** | 0.9652 |
| CD3CN | 422 | **0.0794 (↓ 28.8%)** | 0.1115 | 336 | **0.9443 (↓ 21.1%)** | 1.1973 |
| CD3OD | 253 | **0.1685 (↓ 25.6%)** | 0.2266 | 1,105 | **1.1102 (↓ 23.3%)** | 1.4474 |
| THF-d8 | 38 | **0.1243 (↓ 18.9%)** | 0.1532 | 33 | **0.8342 (↓ 10.1%)** | 0.9281 |
| D2O | 34 | **0.5487 (↓ 3.2%)** | 0.5667 | 144 | **1.5524 (↓ 9.1%)** | 1.7071 |
| DMF-d7 | 20 | **0.0825 (↓ 29.6%)** | 0.1172 | 17 | **0.6098 (↓ 29.6%)** | 0.8667 |
| C2D2Cl4 | 18 | **0.1099 (↓ 1.3%)** | 0.1114 | 10 | **0.5925 (↓ -0.1%)** | 0.5918 |
| PhMe-d8 | 17 | **0.1148 (↓ 48.0%)** | 0.2209 | 12 | **0.9957 (↓ 3.0%)** | 1.0260 |
| CF3CO2D | 11 | **0.4005 (↓ 8.7%)** | 0.4389 | 5 | **1.6959 (↓ 9.1%)** | 1.8652 |
| pyridine-d5 | 6 | **0.2323 (↓ 3.0%)** | 0.2395 | 7 | **2.1323 (↓ -4.6%)** | 2.0393 |
| CD3CO2D | — | — | — | 3 | **1.5534 (↓ 12.0%)** | 1.7645 |
| Not known | 1,852 | **0.0616 (↓ 1.1%)** | 0.0623 | 1,536 | **0.5569 (↓ 1.9%)** | 0.5677 |
| All | 176,985 | **0.0492 (↓ 12.5%)** | 0.0562 | 140,875 | **0.5016 (↓ 5.0%)** | 0.5281 |

analysis: (1) A finer-grained solvent representation further improves the prediction accuracy, reducing the MAE from 0.0501 to 0.0492 for ¹H and from 0.5042 to 0.5016 for ¹³C. (2) CDCl3 shows relatively limited improvement, likely because it dominates the dataset. As a result, models without explicit solvent information implicitly learn a bias toward CDCl3-like environments.
（3）For ¹H shifts, nonpolar solvents with similar properties to CDCl3 (e.g., CD2Cl2 and C2D2Cl4) show relatively small improvements. In contrast, polar or hydrogen-bonding solvents (DMSO-d6, acetone-d6, CD3OD, THF-d8, DMF-d7) and aromatic solvents (C6D6 and PhMe-d8) exhibit larger gains, consistent with chemical intuition that stronger solute–solvent interactions more significantly perturb proton chemical shifts.

We will add these detailed results and analysis in the revised version.


<!-- > **W4 (Minor Soundness).** The \(\lambda\) sweep (Figure 3) uses powers of two from \(2^0\) to \(2^7\); the optimum is \(\lambda=16\) (\(2^4\)). On NMRShiftDB2 (\(L_{\mathrm{atom}}\)), \(^1\)H MAE is ~0.170 at \(\lambda=16\) vs. ~0.190 at \(\lambda=128\); \(^{13}\)C MAE ~0.93 vs. ~1.15 over the same range. A finer grid near the optimum or a sensitivity curve would show how sharp the basin is. Easy to add. -->

#### Response-4 (W4: λ sensitivity)

We **refine** the sweep around the previous optimum (**λ∈[8,32]**); curves are shown in https://anonymous.4open.science/r/NMRNetplusplus-8C70/figure/exp.pdf. Updated minima are λ≈20 (¹H) and λ≈10 (¹³C).

The loss landscape is **flat**: on NMRShiftDB2 (L_atom MAE), ¹H stays **0.170–0.172** for **λ∈[8,28]** and ¹³C stays **0.92–0.94** for **λ∈[4,20]**.


> **W5 (Minor Presentation).** The heteroatom results (Table 4) report MAE values of 2.2809 ppm (¹⁹F), 1.3099 ppm (³¹P), 0.8287 ppm (¹¹B), and 1.9186 ppm (²⁹Si), with R² values of 0.7216, 0.9634, 0.9406, and 0.8901, respectively. However, these are presented without any baselines or comparison methods, making it difficult to contextualize whether these numbers represent strong or weak performance. Even a simple DFT or HOSE comparison would anchor them. The ¹⁹F R² of 0.7216 in particular seems relatively low and warrants discussion.

#### Response-5 (W5 & Q4: heteroatom baselines)

HOSE 方法依赖于数据库，而杂核化学位移数据稀缺，因此此前用HOSE方法预测杂核化学位移的结果比较少. DFT 方法的精度依赖于基组的选择，并且非常耗时，此前没有工作在大规模数据集上做过完整的计算，并以计算简单小分子为主。
我们找到一些已发表的DFT或HOSE方法预测杂核化学位移的结果，希望提供一些comparison参考 in Table C4.

Method|19F MAE|31P MAE|11B MAE|29Si MAE|
|---|---|---|---|---|
|DFT|~5-10(MAE)|~10-15(RMSE)|~3.5(RMSE)|~7.2(MAE)|
|(Ours)|2.28|1.31|0.83|1.92|

[1] Ukhanev S A, Rusakov Y Y, Rusakova I L. A Quest for Effective 19F NMR Spectra Modeling: What Brings a Good Balance Between Accuracy and Computational Cost in Fluorine Chemical Shift Calculations?[J]. International Journal of Molecular Sciences, 2025, 26(14): 6930.
[2] Latypov, Shamil K., et al. "Quantum chemical calculations of 31 P NMR chemical shifts: scopes and limitations." Physical Chemistry Chemical Physics 17.10 (2015): 6976-6987.
[3] Gao, Peng, et al. "11B NMR chemical shift predictions via density functional theory and gauge-including atomic orbital approach: Applications to structural elucidations of boron-containing molecules." ACS omega 4.7 (2019): 12385-12392.
[4] Bursch, Markus, et al. "Comprehensive benchmark study on the calculation of 29Si NMR chemical shifts." Inorganic Chemistry 60.1 (2020): 272-285.



<!-- > **W6 (Minor Presentation).** Wall-clock training time is not reported. Table 7 shows 10 epochs for semi-supervised vs. 50 for supervised, with batch sizes of 4 (labeled) + 16 (unlabeled) for semi-supervised vs. 8 (labeled only) for supervised, and learning rates of 4e-4 vs. 1e-4 respectively. Since the weakly-supervised data contains 898,422 ¹H and 704,373 ¹³C entries compared to NMRShiftDB2's 12,800 ¹H and 26,859 ¹³C entries (~26-70× larger), the actual computational cost is difficult to assess from epoch counts alone. This matters for practitioners deciding whether to adopt the method. -->


#### Response-6 (W6 & Q3: wall-clock training)

On one RTX 4090, supervised (50 epochs, bs 8) takes **2.31 h** (¹H) / **4.58 h** (¹³C); semi-supervised (10 epochs, bs 4+16) takes **24.80 h** (¹H) / **18.34 h** (¹³C)—much larger but completely acceptable. 

We also ran supervised training to **100/200** epochs, and validation **plateaued**, so the **50-epoch** supervised run is **not under-trained**. Both hyperparameters are independently optimized (Appendix D).


<!-- ## Key Questions For Authors

> Can you evaluate on a scaffold-split held-out set from ShiftDB-Lit that is OOD for both the baseline and semi-supervised model? This would directly address W1 and W2 by isolating the semi-supervised learning gain from distribution coverage. -->

<!-- #### Response-7 (Q1: scaffold split)
See Response-1. -->


<!-- > What is the solvent distribution within the “others” category (5,553 \(^1\)H test molecules, 5,485 \(^{13}\)C test molecules), and have you tried finer granularity? The ↓9.5% (\(^1\)H) and ↓9.0% (\(^{13}\)C) improvements for “others” vs. ↓46.8% and ↓17.8% for DMSO-d\(_6\) suggest the single embedding may be underperforming. (Addresses W3.) -->

<!-- #### Response-8 (Q2: solvent “others”)
See Response-2. -->


<!-- > What are the wall-clock training times for the supervised (50 epochs, batch size 8) vs. semi-supervised (10 epochs, batch sizes 4+16) settings on the reported NVIDIA RTX 4090? (Addresses W6.) -->

<!-- #### Response-9 (Q3: wall-clock time)
See Response-6. -->


<!-- > For \(^{19}\)F, the \(R^2\) of 0.7216 is notably lower than for \(^{31}\)P (0.9634) and \(^{11}\)B (0.9406). Can you comment on whether this reflects intrinsic difficulty, data noise, or the broader chemical shift range (~300 to 300 ppm per Table 6)? -->

<!-- #### Response-10 (Q4: ¹⁹F R²)
See Response-5. -->
