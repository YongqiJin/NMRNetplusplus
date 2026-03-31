<!-- # Response to Reviewer 6gyS (review2)

## Weaknesses -->

Thank you for your time and thoughtful reviews! We address your comments point by point as follows.

**("W" refers to Weaknesses, "Q" refers to Key Questions)**


<!-- > No error bars are provided in any of the result tables; it is very hard to tell if the differences are significant or not.


> Many element of Table 2 left unfilled but I fail to see the reason why they cannot be computed. Any prediction method can be applied to the structures in the ShiftDB-Lit database, and the \(L_{\mathrm{mol}}\) loss can be computed on the prediction. I may missing something (in that case please help me here), but I see no reason why the missing values cannot be computed in Table 2. -->


#### Response-1 (W1: error bars; W2: Table 2 blanks)

We thank the reviewer for helping us make Table 2 more complete.

- **Error bars.** We report **mean ± standard deviation** over five random seeds for our semi-supervised NMRNet; variation is **small**, indicating **stable** training.
- **Missing cells.** In the original manuscript, **NMRShiftDB2** was the **primary** benchmark, ShiftDB-Lit **L_mol** columns for baselines mainly used to evaluate the contribution of semi-supervised learning. We agree these metrics are well-defined and computable. We have reproduced all baselines and filled the missing cells.


Table A1: completed Table 2.

- **¹H**

| Method | NMRShiftDB2 (L_atom) MAE | NMRShiftDB2 (L_atom) RMSE | NMRShiftDB2 (L_mol) MAE | NMRShiftDB2 (L_mol) RMSE | ShiftDB-Lit (L_mol) MAE | ShiftDB-Lit (L_mol) RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HOSE | 0.3102 | 0.6587 | 0.2771 | 0.5563 | 0.2159 | 0.3931 |
| GCN | 0.2423 | 0.5491 | 0.2152 | 0.4553 | 0.1712 | 0.3398 |
| FCG | 0.2253 | 0.4914 | 0.2036 | 0.4196 | 0.1562 | 0.2986 |
| SGNN | 0.2152 | 0.4868 | 0.1915 | 0.4028 | 0.1503 | 0.2943 |
| GT-NMR | (0.158)* | (0.293)* | — | — | — | — |
| NMRNet (Baseline) | 0.1972 | 0.4564 | 0.1761 | 0.3896 | 0.1395 | 0.2790 |
| NMRNet (Semi-supervised) | 0.1718 ± 0.0011 | 0.4377 ± 0.0030 | 0.1497 ± 0.0009 | 0.3647 ± 0.0018 | 0.0548 ± 0.0009 | 0.1837 ± 0.0007 |
| ↓ | (12.9%) | (4.1%) | (15.0%) | (6.4%) | (60.7%) | (34.2%) |

\*GT-NMR (¹H) predicts only hydrogens bonded to carbon; full-atom comparison is therefore not applicable.

- **¹³C**

| Method | NMRShiftDB2 (L_atom) MAE | NMRShiftDB2 (L_atom) RMSE | NMRShiftDB2 (L_mol) MAE | NMRShiftDB2 (L_mol) RMSE | ShiftDB-Lit (L_mol) MAE | ShiftDB-Lit (L_mol) RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HOSE | 2.5804 | 4.8495 | 2.3095 | 4.1902 | 2.3753 | 4.4148 |
| GCN | 1.3043 | 2.5103 | 1.1573 | 2.1906 | 1.2551 | 2.9506 |
| FCG | 1.3589 | 2.3487 | 1.2190 | 2.0630 | 1.2620 | 2.8902 |
| SGNN | 1.2606 | 2.2097 | 1.1206 | 1.9138 | 1.2015 | 2.8771 |
| GT-NMR | 1.1647 | 2.1434 | 1.0387 | 1.8651 | 1.1077 | 2.8071 |
| NMRNet (Baseline) | 1.1518 | 2.1398 | 1.0143 | 1.8513 | 1.2591 | 2.9207 |
| NMRNet (Semi-supervised) | 0.9289 ± 0.0019 | 1.9270 ± 0.0115 | 0.7777 ± 0.0013 | 1.5770 ± 0.0127 | 0.5066 ± 0.0014 | 2.3503 ± 0.0011 |
| ↓ | (19.4%) | (9.9%) | (23.3%) | (14.8%) | (59.8%) | (19.5%) |

**¹H GT-NMR** stays **partially** empty: the **original** model targets **H–C** hydrogens only, so it is **not** directly comparable to our full evaluation (footnote).


<!-- > The idea of using a set based supervision loss is not particularly novel. -->

#### Response-2 (W2: novelty of the set loss)

We **agree** that set-based supervision is **not new as a general ML idea**. Our contribution is **not** to propose the abstract notion of “train on unordered sets,” but to **instantiate and scale** it for **NMR chemical-shift prediction from literature spectra** where atom–peak assignment is missing.

Our central claim is that training on large-scale literature spectra without atom-level assignment is effective: on NMRShiftDB2, the gains exceed those obtained from recent architectural changes alone. We emphasize a shift from model-centric to **data-centric AI** for scientific prediction, and view this work as a step toward scalable **scientific data infrastructure**. We will make this framing explicit in the revised introduction and contributions.


<!-- ## Key Questions for Authors -->

<!-- > Please provide error bars for MAE and RMSE values, and provide the missing values in Table 2. -->

#### Response-3 (Q1: error bars)
See Response-1.



<!-- > In general the term “unsupervised” is used many times in the paper, but actually the literature data is not unsupervised as there is a label, a set label. As properly named other places, this is a weakly supervised setting. To avoid confusion please change the occurences of "unsupervised" to "weakly-supervised" in the text. Also semi-supervised can be changed to weakly-supervised. (Except for the title, that you cannot change anymore.)

[Reference Response]
要点:
(1) 完全同意：文献侧是**集合标签**，属于弱监督而非「完全无监督」。文中 **semi-supervised** 专指**原子级标注数据与集合级（无指认）数据联合训练**；**weakly-supervised** 用于仅依赖集合级弱标签的设定，以便二者区分。修改稿中将统一术语并在引言/方法处简要说明。
(2) 我们已全文检索；若当前稿件中仍有 **unsupervised** 残留，将一律改为 **weakly-supervised**（标题除外则于正文与摘要中统一说明）。若审稿人方便标注页码/行号或段落，我们将逐处修订；也欢迎指出我们未意识到的同义表述。 -->

#### Response-4 (Q2: terminology)

We completely agree with the reviewer's comments. We use "semi-supervised" to refer to the training with both atom-level and set-level data, to distinguish it from the "weakly-supervised" setting that only uses set-level data. We will change the occurrences of "unsupervised" to "weakly-supervised" in the text and make it more clear in the revised version.


<!-- > It seems you ignore multiplicity of 1H NMR peaks altogether. This is an easy to use , and valuable information to restrict your possible permutations. Did you tried using it?

[Reference Response]
要点:
(1) 我们认同这是一个很好且专业的想法。对于 $^1$H 谱，我们曾尝试用 multiplicity 约束可行排列，但在 ShiftDB-Lit 上的实验未带来提升。原因主要有二：(i) 据统计，ShiftDB-Lit 中超过 40\% 的 $^1$H 峰标注为 ``m''（multiplet），且存在少量峰型标注错误，会错误地排除部分本应合法的排列；(ii) 该思路依赖由结构对峰型作较可靠预测，而当前实现以规则为主，峰型预测并不完全准确。二者共同限制了 multiplicity 约束的实际收益。
(2) 我们仍认为该方向有价值；若未来峰型预测更准确，或配合更高质量的 NMR 数据（例如原始实验谱图），有希望进一步提升性能。 -->

#### Response-5 (Q3: multiplicity constraint)

Yes, we tried restricting permutations with peak multiplicity for ¹H, but it yielded no gain on ShiftDB-Lit. Two reasons: (i) over 40% of ¹H peaks are labeled "m" (multiplet), and occasional wrong labels can incorrectly exclude valid permutations; (ii) the constraint assumes reliable multiplicity-from-structure prediction, which the current rule-based method does not yet provide. Under these conditions, multiplicity pruning has limited benefit.

We still consider the direction promising—e.g., stronger multiplicity predictors or higher-quality data (including raw spectra) may help.


<!-- > On Figure 3, the model-collapse argument about the red line in the case of \(^1\)H seems a bit fragile if we see that \(^{13}\)C line does not show this behaviour. There is nothing mentioned about this in the text. Please elaborate what could be the possible reason?

[Reference Response]
要点:
(1) 实验现象：Fig.~3 中红线表明，随权重 $\lambda$ 增大，ShiftDB-Lit 上 $L_{\mathrm{mol}}$ 对 $^1$H 几乎单调下降，对 $^{13}$C 则呈 U 形。这表明：对 $^1$H，$\lambda$ 过大会使模型过度拟合 ShiftDB-Lit 的 $L_{\mathrm{mol}}$，而损害原子级正确性；对 $^{13}$C，即使 $\lambda$ 很大也无法使 $L_{\mathrm{mol}}$ 持续下降，少量原子级标注反而能改善 ShiftDB-Lit 上 $L_{\mathrm{mol}}$ 的学习。
(2) 我们猜测可能的原因包括：$^1$H 化学位移范围更窄、更集中，而 $^{13}$C 更宽、更分散。因此，$^1$H 模型更容易在 $L_{\mathrm{mol}}$ 上过拟合；而 $^{13}$C 要在 $L_{\mathrm{mol}}$ 上取得较低损失需要相对正确的指认，因而需要少量原子级标注才能推动 ShiftDB-Lit 上 $L_{\mathrm{mol}}$ 的学习，从而出现 U 形。
(3) 感谢审稿人指出这一点；我们将在修改稿中补充更详细的分析与讨论。 -->

#### Response-6 (Fig.~3: λ vs. L_mol)

In Fig.~3, increasing λ drives ShiftDB-Lit L_mol down almost monotonically for ¹H but yields a U-shaped curve for ¹³C. We interpret this as: for ¹H, large λ favors fitting ShiftDB-Lit’s L_mol over atom-level accuracy; for ¹³C, L_mol does not keep improving with λ alone, and a small amount of atom-level supervision can help L_mol.

We attribute this partly to ¹H shifts spanning a narrower range: many alternative peak-to-atom matchings can yield similarly low L_mol, so that objective is easier to overfit; ¹³C spans a wider range, so low L_mol depends more on correct assignments, yielding the U-shaped curve.

We thank the reviewer and will expand this discussion in the revision.


<!-- > How many conformers do you generate and use as input? -->

#### Response-7 (conformers)

We use **one** conformer per molecule for scalability. On 1,000 molecules, averaging predictions over 10 random conformers changed MAE by **<0.01 ppm** (¹H) and **<0.1 ppm** (¹³C) relative to a single conformer, which is negligible. So we keep single-conformer training and evaluation, and multi-conformer sampling may still help for highly flexible molecules during inference.


<!-- > **Minor:** Using \(\hat{s}\) as ground truth and \(s\) as the prediction goes against traditional use, where a hat denotes the estimate; it would be more appropriate to switch the two notations. -->

#### Response-8 (notations)

Thank the reviewer for catching this. We will adopt the standard convention in the revision: $\{s_i\}$ for ground truth and $\{\hat{s}_i\}$ for predictions.


## Limitations

> No explicit discussion of limitations are provided. One that come into my mind is the difficulty to guess the dominant conformer(s) in the sample, as that can depend also on the solvent for example, furthermore there can be an ensemble of conformers contributing to the signal.


[Response-9] [TODO]
