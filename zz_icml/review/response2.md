# Response to Reviewer 6gyS (review2)

## Weaknesses

> No error bars are provided in any of the result tables; it is very hard to tell if the differences are significant or not.


> Many element of Table 2 left unfilled but I fail to see the reason why they cannot be computed. Any prediction method can be applied to the structures in the ShiftDB-Lit database, and the \(L_{\mathrm{mol}}\) loss can be computed on the prediction. I may missing something (in that case please help me here), but I see no reason why the missing values cannot be computed in Table 2.

[Response-1]
We thank the reviewer for the suggestions on making Table 2 more complete and persuasive.

For error bars, we repeated our method with five random seeds and report mean ± standard deviation; variation across seeds is small, indicating stable training.

For previously missing entries, we reproduced all methods listed in Table 2 under the same evaluation protocol and completed the remaining metrics. The full results are given in Table B1 and will be integrated into the main text in the revised manuscript.


> The idea of using a set based supervision loss is not particularly novel.

[Response-2]



## Key Questions for Authors

> Please provide error bars for MAE and RMSE values, and provide the missing values in Table 2.

[Response-3]
See Response-1.


> In general the term “unsupervised” is used many times in the paper, but actually the literature data is not unsupervised as there is a label, a set label. As properly named other places, this is a weakly supervised setting. To avoid confusion please change the occurences of "unsupervised" to "weakly-supervised" in the text. Also semi-supervised can be changed to weakly-supervised. (Except for the title, that you cannot change anymore.)

[Reference Response]
要点:
(1) 完全同意：文献侧是**集合标签**，属于弱监督而非「完全无监督」。文中 **semi-supervised** 专指**原子级标注数据与集合级（无指认）数据联合训练**；**weakly-supervised** 用于仅依赖集合级弱标签的设定，以便二者区分。修改稿中将统一术语并在引言/方法处简要说明。
(2) 我们已全文检索；若当前稿件中仍有 **unsupervised** 残留，将一律改为 **weakly-supervised**（标题除外则于正文与摘要中统一说明）。若审稿人方便标注页码/行号或段落，我们将逐处修订；也欢迎指出我们未意识到的同义表述。

[Response-4]
We completely agree with the reviewer's comments. We use "semi-supervised" to refer to the training with both atom-level and set-level data, to distinguish it from the "weakly-supervised" setting that only uses set-level data. We will change the occurrences of "unsupervised" to "weakly-supervised" in the text and explain the difference in the introduction and methods section.

If there are any remaining "unsupervised" in the current version, we will change it to "weakly-supervised" (except for the title, which cannot be changed anymore). If the reviewer can provide the page number/line number or paragraph, we will revise it accordingly. We also welcome the reviewer to point out any other synonymous expressions that we may have missed.


> It seems you ignore multiplicity of 1H NMR peaks altogether. This is an easy to use , and valuable information to restrict your possible permutations. Did you tried using it?

[Reference Response]
要点:
(1) 我们认同这是一个很好且专业的想法。对于 $^1$H 谱，我们曾尝试用 multiplicity 约束可行排列，但在 ShiftDB-Lit 上的实验未带来提升。原因主要有二：(i) 据统计，ShiftDB-Lit 中超过 40\% 的 $^1$H 峰标注为 ``m''（multiplet），且存在少量峰型标注错误，会错误地排除部分本应合法的排列；(ii) 该思路依赖由结构对峰型作较可靠预测，而当前实现以规则为主，峰型预测并不完全准确。二者共同限制了 multiplicity 约束的实际收益。
(2) 我们仍认为该方向有价值；若未来峰型预测更准确，或配合更高质量的 NMR 数据（例如原始实验谱图），有希望进一步提升性能。

[Response-5]
We appreciate the valuable and professional suggestion. For $^1$H, we explored using multiplicity to restrict feasible permutations, but experiments on ShiftDB-Lit did not get better results. We attribute this to two factors: (i) in our statistics, over 40\% of $^1$H peaks are annotated as ``m'' (multiplet) in ShiftDB-Lit, and there are a small number of incorrect multiplicity labels, which can wrongly rule out valid permutations; (ii) this approach relies on reasonably accurate multiplicity prediction from the structure, whereas the current method is largely rule-based and multiplicity estimates are not fully reliable. Together, these limitations cap the practical benefit of multiplicity-based constraints.

We still view this as a promising idea. It may promote the performance of the model when stronger multiplicity models and higher-quality NMR data (e.g., raw experimental spectra) are used.


> On Figure 3, the model-collapse argument about the red line in the case of \(^1\)H seems a bit fragile if we see that \(^{13}\)C line does not show this behaviour. There is nothing mentioned about this in the text. Please elaborate what could be the possible reason?

[Reference Response]
要点:
(1) 实验现象：Fig.~3 中红线表明，随权重 $\lambda$ 增大，ShiftDB-Lit 上 $L_{\mathrm{mol}}$ 对 $^1$H 几乎单调下降，对 $^{13}$C 则呈 U 形。这表明：对 $^1$H，$\lambda$ 过大会使模型过度拟合 ShiftDB-Lit 的 $L_{\mathrm{mol}}$，而损害原子级正确性；对 $^{13}$C，即使 $\lambda$ 很大也无法使 $L_{\mathrm{mol}}$ 持续下降，少量原子级标注反而能改善 ShiftDB-Lit 上 $L_{\mathrm{mol}}$ 的学习。
(2) 我们猜测可能的原因包括：$^1$H 化学位移范围更窄、更集中，而 $^{13}$C 更宽、更分散。因此，$^1$H 模型更容易在 $L_{\mathrm{mol}}$ 上过拟合；而 $^{13}$C 要在 $L_{\mathrm{mol}}$ 上取得较低损失需要相对正确的指认，因而需要少量原子级标注才能推动 ShiftDB-Lit 上 $L_{\mathrm{mol}}$ 的学习，从而出现 U 形。
(3) 感谢审稿人指出这一点；我们将在修改稿中补充更详细的分析与讨论。

[Response-6]
We first summarize the red curves in Fig.~3: as the weight $\lambda$ increases, $L_{\mathrm{mol}}$ on ShiftDB-Lit decreases almost monotonically for $^1$H, whereas for $^{13}$C it follows a U-shaped trend. This pattern suggests that for $^1$H, overly large $\lambda$ encourages fitting ShiftDB-Lit’s $L_{\mathrm{mol}}$ at the expense of atom-level correctness. For $^{13}$C, even very large $\lambda$ does not make $L_{\mathrm{mol}}$ keep decreasing; a small amount of atom-level supervision can instead improve learning of $L_{\mathrm{mol}}$ on ShiftDB-Lit.

We hypothesize that this difference is related to the fact that $^1$H chemical shifts span a narrower, more concentrated range, whereas $^{13}$C shifts are spread over a wider range. Consequently, the $^1$H model is more prone to overfitting under $L_{\mathrm{mol}}$, while $^{13}$C requires relatively correct assignments to achieve low $L_{\mathrm{mol}}$, so a small amount of atom-level labels helps and produces the U-shaped curve.

We thank the reviewer for raising this point and will add a more detailed discussion in the revised manuscript.


> How many conformers do you generate and use as input?

[Response-7]
We use a single conformer as input, which keeps the pipeline efficient and scalable for large datasets during both training and inference.

To assess the effect of conformational sampling, we conducted an experiment on 1,000 molecules where 10 random conformers were generated for each molecule and the predicted shifts were averaged. The difference compared to using a single conformer was negligible (average change <0.01 ppm for ¹H and <0.1 ppm for ¹³C).

Therefore, we believe that using a single conformer is sufficient for large-scale training and evaluation. For practical applications involving highly flexible molecules, sampling multiple conformers could further improve prediction accuracy.


> **Minor:** Using \(\hat{s}\) as ground truth and \(s\) as the prediction goes against traditional use, where a hat denotes the estimate; it would be more appropriate to switch the two notations.

[Response-8]
We thank the reviewer for the suggestion. We will switch the notations of the ground truth and the prediction in the text, using \(\{s_i\}\) to represent the ground truth and \(\{\hat{s}_i\}\) to represent the prediction.


## Limitations

> No explicit discussion of limitations are provided. One that come into my mind is the difficulty to guess the dominant conformer(s) in the sample, as that can depend also on the solvent for example, furthermore there can be an ensemble of conformers contributing to the signal.


[Response-9] [TODO]
