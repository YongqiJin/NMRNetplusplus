# Response to Reviewer wmHV (review3)

## Weaknesses (W1–W6)

> **W1 (Major Soundness).** The ShiftDB-Lit test set evaluation conflates two effects. The NMRNet baseline is trained only on NMRShiftDB2, making ShiftDB-Lit out-of-distribution for it, while the semi-supervised model trains on ShiftDB-Lit, making it in-distribution. The reported reductions ¹H MAE from 0.1395 to 0.0559 (↓59.9%) and ¹³C MAE from 1.2591 to 0.5060 (↓59.8%) therefore reflect both the value of semi-supervised learning and simple distribution coverage. The authors acknowledge this (lines 283–293) but do not attempt to disentangle the two contributions for instance, by evaluating on a held-out scaffold split that is OOD for both models. This is fixable: a scaffold-based partition of ShiftDB-Lit that excludes training scaffolds from both models would isolate the semi-supervised learning gain.


> **W2 (Major Soundness).** ShiftDB-Lit uses a random 4:1 train/test split (Section 4.1) rather than a scaffold split. For a dataset of ~1.6M molecules, random splitting almost certainly places structurally similar molecules on both sides of the partition, inflating reported metrics. The NMRShiftDB2 results use a pre-defined benchmark split and are less affected, but the ShiftDB-Lit numbers which show the most dramatic gains (59.9% and 59.8% MAE reductions) and carry the paper's strongest narrative claims are the most vulnerable to this concern. This is also fixable by re-evaluating with a scaffold split.

[Response-1]
We thank the reviewer for raising this issue and giving us the constructive suggestion. We acknowledge that the random 4:1 train/test split in ShiftDB-Lit can allow leakage of structurally similar molecules across the boundary. To better disentangle the two contributions, we further split the ShiftDB-Lit test molecules by their maximum Tanimoto similarity (2048-bit Morgan fingerprints) to the union of training structures from NMRShiftDB2 and ShiftDB-Lit: similar ($\geq 0.7$), intermediate ($0.5$–$0.7$), and dissimilar ($< 0.5$).

The results are summarized in Table C1. On the similar test subset, the semi-supervised model improves strongly, consistent with reducing an OOD-vs-ID gap by better covering the training distribution. On the dissimilar subset, the semi-supervised model still significantly outperforms the baseline, but the gain is less pronounced than on the similar subset.

[Table C1]

We will add this detailed analysis to the revised manuscript.


> **W3 (Major Soundness).** The solvent embedding is limited to three categories: CDCl₃ (89.1% of data, 162,509 test molecules), DMSO-d₆ (5.7%, 11,623 test molecules), and "others" (5,553 test molecules for ¹H). Table 3 shows the catch-all "others" category retains the highest MAE (0.0996 for ¹H, 0.8684 for ¹³C with solvent incorporation), and its improvement over the solvent-agnostic baseline is modest (↓9.5% for ¹H, ↓9.0% for ¹³C) compared to the gains for DMSO-d₆ (↓46.8% for ¹H, ↓17.8% for ¹³C). There is no analysis of (a) whether the "others" embedding does meaningful work beyond what the model learns without it, or (b) whether finer solvent granularity (e.g., 5–10 categories) would improve performance. Given that ShiftDB-Lit presumably contains solvent identity metadata beyond these three bins, the coarse grouping appears to be an underexplored design choice rather than a hard constraint. A targeted ablation comparing 3-bin vs. finer-grained solvent embeddings would address this.

[Response-2] [TODO]

[Table C2]



> **W4 (Minor Soundness).** The \(\lambda\) sweep (Figure 3) uses powers of two from \(2^0\) to \(2^7\); the optimum is \(\lambda=16\) (\(2^4\)). On NMRShiftDB2 (\(L_{\mathrm{atom}}\)), \(^1\)H MAE is ~0.170 at \(\lambda=16\) vs. ~0.190 at \(\lambda=128\); \(^{13}\)C MAE ~0.93 vs. ~1.15 over the same range. A finer grid near the optimum or a sensitivity curve would show how sharp the basin is. Easy to add.

[Response-3]
We thank the reviewer for this suggestion. We add a finer grid near the previous optimum ($\lambda \in [8,32]$) and report the results in [Figure Link](https://anonymous.4open.science/r/NMRNetplusplus-8C70/figure/exp.pdf). The refined optimum is near 20 for $^1$H and 10 for $^{13}$C.

The basin is flat: for $^1$H, $L_{\mathrm{atom}}$ MAE on NMRShiftDB2 stays between 0.170 and 0.172 for $\lambda \in [8,28]$; for $^{13}$C, MAE stays between 0.92 and 0.94 for $\lambda \in [4,20]$.


> **W5 (Minor Presentation).** The heteroatom results (Table 4) report MAE values of 2.2809 ppm (¹⁹F), 1.3099 ppm (³¹P), 0.8287 ppm (¹¹B), and 1.9186 ppm (²⁹Si), with R² values of 0.7216, 0.9634, 0.9406, and 0.8901, respectively. However, these are presented without any baselines or comparison methods, making it difficult to contextualize whether these numbers represent strong or weak performance. Even a simple DFT or HOSE comparison would anchor them. The ¹⁹F R² of 0.7216 in particular seems relatively low and warrants discussion.

[Response-4]

HOSE 方法依赖于数据库，而杂核化学位移数据稀缺，因此此前用HOSE方法预测杂核化学位移的结果比较少. DFT 方法的精度依赖于基组的选择，并且非常耗时，此前没有工作在大规模数据集上做过完整的计算，并以计算简单小分子为主。
我们找到一些已发表的DFT或HOSE方法预测杂核化学位移的结果，希望提供一些comparison参考 in Table C4.

Method | 19F MAE | 31P MAE | 11B MAE | 29Si MAE |
| --- | --- | --- | --- | --- |
| DFT | ~5-10(MAE) | ~10-15(RMSE) | ~3.5(RMSE) | ~7.2(MAE) |
| (Ours) | 2.28 | 1.31 | 0.83 | 1.92 |

[1] Ukhanev S A, Rusakov Y Y, Rusakova I L. A Quest for Effective 19F NMR Spectra Modeling: What Brings a Good Balance Between Accuracy and Computational Cost in Fluorine Chemical Shift Calculations?[J]. International Journal of Molecular Sciences, 2025, 26(14): 6930.
[2] Latypov, Shamil K., et al. "Quantum chemical calculations of 31 P NMR chemical shifts: scopes and limitations." Physical Chemistry Chemical Physics 17.10 (2015): 6976-6987.
[3] Gao, Peng, et al. "11B NMR chemical shift predictions via density functional theory and gauge-including atomic orbital approach: Applications to structural elucidations of boron-containing molecules." ACS omega 4.7 (2019): 12385-12392.
[4] Bursch, Markus, et al. "Comprehensive benchmark study on the calculation of 29Si NMR chemical shifts." Inorganic Chemistry 60.1 (2020): 272-285.



> **W6 (Minor Presentation).** Wall-clock training time is not reported. Table 7 shows 10 epochs for semi-supervised vs. 50 for supervised, with batch sizes of 4 (labeled) + 16 (unlabeled) for semi-supervised vs. 8 (labeled only) for supervised, and learning rates of 4e-4 vs. 1e-4 respectively. Since the weakly-supervised data contains 898,422 ¹H and 704,373 ¹³C entries compared to NMRShiftDB2's 12,800 ¹H and 26,859 ¹³C entries (~26-70× larger), the actual computational cost is difficult to assess from epoch counts alone. This matters for practitioners deciding whether to adopt the method.


[Response-5]
We add wall-clock training times for the supervised (50 epochs, batch size 8) and semi-supervised (10 epochs, batch sizes 4+16) configurations in Table C3.

Table C3: Wall-clock time (hours, a single NVIDIA RTX 4090).
| Method | ¹H | ¹³C |
| --- | ---: | ---: |
| Supervised | 2.31 | 4.58 |
| Semi-supervised | 24.80 | 18.34 |

Semi-supervised training draws on a much larger dataset, so it requires more wall-clock time than supervised training, but 24-hour wall-clock time on a single NVIDIA RTX 4090 is totally acceptable.

Because the two settings use different training budgets, we also ran supervised training for 100 and 200 epochs; validation metrics plateaued, so the 50-epoch supervised run is not under-trained. The 50-epoch (supervised) and 10-epoch (semi-supervised) schedules are both selected from validation performance (Appendix D).


## Key Questions For Authors

> Can you evaluate on a scaffold-split held-out set from ShiftDB-Lit that is OOD for both the baseline and semi-supervised model? This would directly address W1 and W2 by isolating the semi-supervised learning gain from distribution coverage.

[Response-6]
See Response-1.


> What is the solvent distribution within the “others” category (5,553 \(^1\)H test molecules, 5,485 \(^{13}\)C test molecules), and have you tried finer granularity? The ↓9.5% (\(^1\)H) and ↓9.0% (\(^{13}\)C) improvements for “others” vs. ↓46.8% and ↓17.8% for DMSO-d\(_6\) suggest the single embedding may be underperforming. (Addresses W3.)

[Response-7]
See Response-2.


> What are the wall-clock training times for the supervised (50 epochs, batch size 8) vs. semi-supervised (10 epochs, batch sizes 4+16) settings on the reported NVIDIA RTX 4090? (Addresses W6.)

[Response-8]
See Response-5.


> For \(^{19}\)F, the \(R^2\) of 0.7216 is notably lower than for \(^{31}\)P (0.9634) and \(^{11}\)B (0.9406). Can you comment on whether this reflects intrinsic difficulty, data noise, or the broader chemical shift range (~300 to 300 ppm per Table 6)?

[Response-9]
See Response-4.
