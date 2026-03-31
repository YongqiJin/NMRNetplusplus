# Response to Reviewer iTNC (review1)

## Weaknesses / concerns

> My main concern is evaluation fairness and interpretation. The paper emphasizes large gains on ShiftDB-Lit, but the baseline is trained only on NMRShiftDB2 while the proposed model leverages ShiftDB-Lit during training, so the comparison is partly OOD-vs-ID rather than purely method-vs-method. The paper does acknowledge this, but the framing should be more careful.

[Reference Response]
要点:
(1) 承认真是一个 OOD vs ID 的问题。paper 承认了 nmrshiftdb2 更适合作为公认的主要的基准，在 Shiftdb-Lit 上的结果主要为了说明，半监督训练在当前的带指认数据有限的情况下，极大的扩充模型的化学空间，反映了落地场景下的应用范围的扩大。我们也会更注意措辞说明这一点，避免将 Shiftdb-Lit 上的结果单独作为方法的贡献。
(2) 在原文的 Table 5 中，我们做了关于数据集的消融实验, 结果说明在同一个有监督的数据集上，"augmenting supervised training with an additional weakly-supervised loss does not improve performance", 因此半监督训练的贡献在于让无标注的数据也能够得以训练，因此训练方法和数据集的贡献是耦合的。
(3) Furthermore, 为了attempt to disentangle the two contributions, 我们evaluated the performance on a held-out scaffold split that is OOD for both models, as the third reviewer (wmHV) suggested. 具体结果见 `exp.md` 的 Table 4.

[Response-1]

We thank the reviewer for this careful point. We agree that comparing the proposed model with the baseline on the ShiftDB-Lit test set mixes two contributions: (i) gains from the learning objective, and (ii) a train/test distribution shift (roughly OOD vs. ID). Table 5 shows that when both are trained on the same atom-assigned corpus, augmenting supervised training with an additional weakly supervised loss does not improve performance; the benefit therefore comes from training on unassigned data at scale, so these two contributions are coupled. As we already argue in the paper, NMRShiftDB2 is the primary benchmark for fair comparison; ShiftDB-Lit results are intended to show that semi-supervised training substantially widens chemical coverage when high-quality atom assignments are scarce—reflecting broader, deployment-relevant coverage rather than a standalone claim about the method alone. We will make this framing clearer in the revision.

Furthermore, to disentangle these two factors and following Reviewer wmHV’s suggestion, we additionally evaluate both models on a held-out subset that is structurally OOD with respect to the union of the NMRShiftDB2 and ShiftDB-Lit training scaffolds. These results separate “seeing ShiftDB-Lit during training” from the headline random-split comparison and show that semi-supervised training still improves the supervised baseline under this stricter test. Full experimental details and results are given in Table 4 of our response to Reviewer wmHV (Sec. 4 of our experimental appendix).


> A second concern is novelty level. The application is valuable, and the sorting reduction is elegant, but the overall method is still a relatively direct semi-supervised extension of an existing backbone rather than a fundamentally new model family. The strongest novelty is really the formulation plus the scale of the data resource.

[Reference Response]
要点:
(1) 文章核心贡献在于：论证以大规模、无原子级指认的文献谱作为训练信号的有效性；在 NMRShiftDB2 基准上的收益高于近年来单靠模型结构改进带来的典型增益，并在化学空间覆盖、溶剂效应、多核种拓展等方面体现优势，从而凸显传统纯监督设定下的数据瓶颈。
(2) 本文的主要创新不在新模型结构上，而是希望将讨论重心从 Model-centric AI 转向 Data-centric AI：为 AI for science 提供一个切入点，指向未来科学数据基础设施的建设，以及面向弱标注科学数据的训练范式研究。

[Response-2]

We agree that our strongest novelty lies in the learning formulation and data resource rather than a fundamentally new model family. Our central claim is that training with large-scale literature spectra without atom-level correspondence is effective: on the standard NMRShiftDB2 benchmark, the improvement exceeds what is typically achieved by recent architectural changes alone, and we further demonstrate advantages in chemical-space coverage, solvent-aware modeling, and heteronuclear settings—together highlighting bottlenecks of purely supervised pipelines when high-quality annotations are limited.

The main contribution is not a new architecture but a shift of emphasis from model-centric to data-centric AI for scientific prediction. We position this work as a step toward scalable scientific data infrastructure and learning from weakly labeled scientific data at scale, and we will make this framing explicit in the revised introduction and contributions.


> A third concern is data quality / noise robustness. The literature-extracted dataset is large, but inevitably noisy. The paper describes filtering procedures, which is good, but the main text does not quantify error rates from extraction, OCSR, parsing, solvent normalization, or duplicate handling. Since the paper’s central claim relies on learning from noisy literature data, this deserves more explicit auditing.

[Reference Response]
要点: 
(1) 如 reviewer 所指，literature-extracted dataset is inevitably noisy，这就是为什么我们加了严格的多步筛选来保证数据的质量。To evaluate the quality of the extracted data, we randomly sampled 300 entries from the dataset and manually cross-checked each against 真实的化学位移数据。根据我们的人工校验结果，H谱化学位移的误差1H: MAE 0.026，13C: MAE 0.206。consistent with expectations for experimental NMR data [1]，并且These values are substantially lower than those reported for NMRShiftDB (0.09 ppm for 1H shift and 0.51 ppm for 13C shift) [2]. 数据质量是可以保证的。
[1] Jonas, E. & Kuhn, S. Rapid prediction of NMR spectral properties with quantified uncertainty. J. Cheminform. 11, 50, https://doi. org/10.1186/s13321-019-0374-3 (2019).
[2] Kuhn, S., Kolshorn, H., Steinbeck, C. & Schlörer, N. Twenty years of nmrshiftdb2: A case study of an open database for analytical chemistry. Magn. Reson. Chem. 62, 74–83, https://doi.org/10.1002/mrc.5418 (2024).
(2) 文章中提到的 "noisy", 出现在关于权重 $\lambda$ 的消融实验 (Line 373) 中. 这里指的是弱监督loss使用的是"伪标签"，这会带来"noise", 因此需要调整权重 $\lambda$ 来平衡方差. 这一点是我们表述的不够清楚，我们会在修改稿中指明这一点.

[Response-3]

We thank the reviewer for this important point. The literature-extracted dataset is inevitably noisy. Our multi-step filtering process (Section 3.1) is designed to mitigate this; we manually audited 300 random samples and found H-NMR MAE 0.026 ppm and C-NMR MAE 0.206 ppm, consistent with experimental NMR noise expectations [1] and lower than reported values for NMRShiftDB [2]. Data quality is ensured.
[1] Jonas, E. & Kuhn, S. Rapid prediction of NMR spectral properties with quantified uncertainty. J. Cheminform. 11, 50, https://doi. org/10.1186/s13321-019-0374-3 (2019).
[2] Kuhn, S., Kolshorn, H., Steinbeck, C. & Schlörer, N. Twenty years of nmrshiftdb2: A case study of an open database for analytical chemistry. Magn. Reson. Chem. 62, 74–83, https://doi.org/10.1002/mrc.5418 (2024).

The “noisy” reference in the ablation section (Line 373) refers to the use of “pseudo-labels” for weak supervision, which introduces uncertainty. The hyperparameter \(\lambda\) balances this by adjusting the relative importance of atom-level versus molecular-level objectives; in the revision, we will clarify this in the text.


> The solvent modeling is promising but still fairly coarse. Solvents are grouped into three categories, with “others” collapsed into one embedding. That is understandable for data imbalance reasons, but it limits chemical interpretability and may hide meaningful solvent-specific behavior.

[Reference Response]
正文已说明三分类动机：\ce{CDCl3} 与 \ce{DMSO-d6} 为主流，其余合并为 “others” 以应对**长尾与数据稀疏**（约 165–166 行）。我们同意细粒度溶剂可能带来更可解释或更优的嵌入；可在修改稿中补充 “others” 内分布描述，并视 rebuttal 篇幅讨论是否增加**细粒度溶剂 embedding 消融**。

[TODO: response]

[TODO: Exp]
细粒度溶剂：例如将 “others” 中高频溶剂单独设类或 5–10 类与三桶对比（与 Review 1 Q5 联动）。


> Finally, the paper would benefit from stronger baselines in the weak-supervision setting. Most comparisons are against older supervised predictors or the NMRNet baseline. There is less discussion of alternative set-prediction / permutation-invariant training objectives or semi-supervised baselines beyond the chosen formulation.

[Reference Response]
要点:
(1) 关于 alternative set-prediction / permutation-invariant objectives：我们采用的 bipartite matching 是AI中常用的，与「每个活性原子对应一个化学位移」的一一对应设定一致，在给定指认的情况下能退化到 Supervised loss；它包含了一些常用的set-prediction loss, 比如 Wasserstein loss (OT)，就是MAE版本的 bipartite matching loss。我们用 MAE、MSE、Huber损失函数做过对比，评测结果接近。
(2) 关于 semi-supervised baselines：我们补充了预训练 + 微调与联合半监督训练等策略的对比，并报告不同数据组合；具体设置与结果见我们对 Reviewer pNG7（Review 4）的回复。

[Response-5]

We thank the reviewer for this suggestion.

(1) On alternative set-prediction / permutation-invariant objectives: we use bipartite matching, which is standard in machine learning and matches the one-to-one pairing of active nuclei with chemical shifts. When peak assignments are fixed, the objective reduces to a supervised per-atom loss. This family includes common set-prediction objectives—for example, Wasserstein-1 distance (OT) is the MAE version of bipartite matching loss under 1D discrete setting. We compared MAE, MSE, and Huber as the pairwise regression term, and observed similar evaluation performance.

(2) On semi-supervised baselines: we compare pretrain–finetune with joint semi-supervised training and report results under several data combinations; protocols and numbers are given in our response to Reviewer pNG7 (Review 4).


## Key Questions for Authors

> Can the authors quantify the noise level of ShiftDB-Lit more directly, for example by manually auditing a random subset of extracted examples?

[Response-6]
See Response-3.


> How much of the gain comes from the sorting-based weak loss itself versus simply exposing the model to much broader chemical coverage?

[Response-7]
See Response-1.


> Did the authors deduplicate near-identical molecules or control for overlap between literature-derived molecules and benchmark molecules?

[Response-8]
We removed identical molecules based on canonical SMILES and did not perform additional deduplication for near-identical molecules with different SMILES.

We verified that there is no overlap between the literature-derived spectra and the benchmark spectra, and no overlap between the training and test sets in ShiftDB-Lit.


> Could the authors compare against a Hungarian-loss implementation directly on a smaller subset to verify that the sorting surrogate is empirically equivalent in practice, not only theoretically?

[Response-9]
We appreciate this suggestion of experimental validation. As a simpler, easy-to-reproduce check, we randomly generated 1,000 set pairs and computed the loss using both a Hungarian implementation and the sorting-based implementation. In every case the two losses matched. Code is given below.

See `zz_icml/review/code.py` (run: `python zz_icml/review/code.py`).


> For solvent modeling, what happens if the most common additional solvents are modeled separately rather than merged into “others”?

[Response-10]
See Response-4.


## Limitations

> The paper would be stronger with one extra section on data auditing: extraction quality, solvent label normalization, duplicates, and error examples. It would also help to separate the claims more clearly:
> - “semi-supervised objective is effective,”
> - “large-scale literature data improves coverage,”
> - “solvent conditioning adds value.”

> Right now these are somewhat intertwined. I would also like a clearer comparison to other possible permutation-invariant objectives and perhaps a discussion of when the sorting reduction fails if the loss assumptions are violated.

[Reference Response]


[TODO: response]

