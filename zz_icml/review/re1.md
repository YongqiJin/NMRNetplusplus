<!-- # Response to Reviewer iTNC (review1)

## Weaknesses / concerns -->

Thank you for your time and thoughtful reviews! We address your comments point by point as follows.

**("W" refers to Weaknesses, "Q" refers to Key Questions)**


<!-- > My main concern is evaluation fairness and interpretation. The paper emphasizes large gains on ShiftDB-Lit, but the baseline is trained only on NMRShiftDB2 while the proposed model leverages ShiftDB-Lit during training, so the comparison is partly OOD-vs-ID rather than purely method-vs-method. The paper does acknowledge this, but the framing should be more careful. -->

#### Response-1 (W1: fairness / interpretation)

We agree the ShiftDB-Lit headline comparison is partly OOD vs. ID, not a pure method-only fair test. **NMRShiftDB2** remains the **primary** benchmark; **ShiftDB-Lit** highlights how semi-supervised training **expands chemical-space coverage** beyond fully labeled regimes—closer to **deployment settings** where atom-level assignments are limited.

The ablation in Table 5 shows that **adding the weak loss alone** does not improve performance on the **same** atom-assigned training data; its role is to **enable learning from a much larger assignment-free corpus**. Thus the **weak-loss objective** and **data coverage** are **coupled** contributions. Following Reviewer wmHV’s suggestion, we additionally run a **structure-based split** for a fairer OOD evaluation; details are in **Table A3** (Reviewer wmHV’s Response-1).


<!-- > A second concern is novelty level. The application is valuable, and the sorting reduction is elegant, but the overall method is still a relatively direct semi-supervised extension of an existing backbone rather than a fundamentally new model family. The strongest novelty is really the formulation plus the scale of the data resource.

[Reference Response]
要点:
(1) 文章核心贡献在于：论证以大规模、无原子级指认的文献谱作为训练信号的有效性；在 NMRShiftDB2 基准上的收益高于近年来单靠模型结构改进带来的典型增益，并在化学空间覆盖、溶剂效应、多核种拓展等方面体现优势，从而凸显传统纯监督设定下的数据瓶颈。
(2) 本文的主要创新不在新模型结构上，而是希望将讨论重心从 Model-centric AI 转向 Data-centric AI：为 AI for science 提供一个切入点，指向未来科学数据基础设施的建设，以及面向弱标注科学数据的训练范式研究。-->


#### Response-2 (W2: novelty)

We agree that the strongest novelty lies in the learning formulation and data resource rather than in a fundamentally new architecture. Our central claim is that training on large-scale literature spectra without atom-level assignment is effective: on NMRShiftDB2, the gains exceed those obtained from recent architectural changes alone. And we further show benefits in broader chemical coverage, solvent-aware modeling, and heteronuclear settings.

In the paper, we emphasize a shift from model-centric to **data-centric AI** for scientific machine learning, and view this work as a step toward scalable scientific data infrastructure and large-scale learning from weakly labeled scientific data. We will make this framing explicit in the revised introduction and contributions.



<!-- > A third concern is data quality / noise robustness. The literature-extracted dataset is large, but inevitably noisy. The paper describes filtering procedures, which is good, but the main text does not quantify error rates from extraction, OCSR, parsing, solvent normalization, or duplicate handling. Since the paper’s central claim relies on learning from noisy literature data, this deserves more explicit auditing.

[Reference Response]
要点: 
(1) 如 reviewer 所指，literature-extracted dataset is inevitably noisy，这就是为什么我们加了严格的多步筛选来保证数据的质量。To evaluate the quality of the extracted data, we randomly sampled 300 entries from the dataset and manually cross-checked each against 真实的化学位移数据。根据我们的人工校验结果，H谱化学位移的误差1H: MAE 0.026，13C: MAE 0.206。consistent with expectations for experimental NMR data [1]，并且These values are substantially lower than those reported for NMRShiftDB (0.09 ppm for 1H shift and 0.51 ppm for 13C shift) [2]. 数据质量是可以保证的。
[1] Jonas, E. & Kuhn, S. Rapid prediction of NMR spectral properties with quantified uncertainty. J. Cheminform. 11, 50, https://doi. org/10.1186/s13321-019-0374-3 (2019).
[2] Kuhn, S., Kolshorn, H., Steinbeck, C. & Schlörer, N. Twenty years of nmrshiftdb2: A case study of an open database for analytical chemistry. Magn. Reson. Chem. 62, 74–83, https://doi.org/10.1002/mrc.5418 (2024).
(2) 文章中提到的 "noisy", 出现在关于权重 $\lambda$ 的消融实验 (Line 373) 中. 这里指的是弱监督loss使用的是"伪标签"，这会带来"noise", 因此需要调整权重 $\lambda$ 来平衡方差. 这一点是我们表述的不够清楚，我们会在修改稿中指明这一点. -->


#### Response-3 (W3: data quality / noise robustness)

We thank the reviewer for this important point. The literature-extracted dataset is inevitably noisy. Our multi-step filtering process is designed to mitigate this. We **manually audited** 300 random samples and found **¹H MAE 0.026 ppm and ¹³C MAE 0.206 ppm**, consistent with experimental noise expectations, so data quality is ensured.

The “noisy” reference in the ablation section (Line 373) refers to the use of “pseudo-labels” for weak supervision, which introduces uncertainty. The hyperparameter λ balances this by adjusting the relative importance of atom-level versus molecular-level objectives. We will clarify this in the revision.


<!-- > The solvent modeling is promising but still fairly coarse. Solvents are grouped into three categories, with “others” collapsed into one embedding. That is understandable for data imbalance reasons, but it limits chemical interpretability and may hide meaningful solvent-specific behavior.

[Reference Response]
正文已说明三分类动机：\ce{CDCl3} 与 \ce{DMSO-d6} 为主流，其余合并为 “others” 以应对**长尾与数据稀疏**（约 165–166 行）。我们同意细粒度溶剂可能带来更可解释或更优的嵌入；可在修改稿中补充 “others” 内分布描述，并视 rebuttal 篇幅讨论是否增加**细粒度溶剂 embedding 消融**。 -->

#### Response-4 (W4: fine-grained solvent embedding)

We thank the reviewer for raising this point. The original solvent grouping was coarse; we **disaggregated** the **“others”** bucket into the **15 distinct solvent labels** present in the dataset, with detailed experiments and analysis in **Table A3** and **Reviewer wmHV's Response-3**.


<!-- > Finally, the paper would benefit from stronger baselines in the weak-supervision setting. Most comparisons are against older supervised predictors or the NMRNet baseline. There is less discussion of alternative set-prediction / permutation-invariant training objectives or semi-supervised baselines beyond the chosen formulation.

[Reference Response]
要点:
(1) 关于 alternative set-prediction / permutation-invariant objectives：我们采用的 bipartite matching 是AI中常用的，与「每个活性原子对应一个化学位移」的一一对应设定一致，在给定指认的情况下能退化到 Supervised loss；它包含了一些常用的set-prediction loss, 比如 Wasserstein loss (OT)，就是MAE版本的 bipartite matching loss。我们用 MAE、MSE、Huber损失函数做过对比，评测结果接近。
(2) 关于 semi-supervised baselines：我们补充了预训练 + 微调与联合半监督训练等策略的对比，并报告不同数据组合；具体设置与结果见我们对 Reviewer pNG7（Review 4）的回复。 -->

#### Response-5 (W5: weak-supervision baselines)

(1) Alternative set-prediction objectives
We use bipartite matching, which is standard in machine learning, for the one-to-one pairing of active nuclei with chemical shifts. When peak assignments are fixed, the objective reduces to a supervised per-atom loss. This family includes common set-prediction objectives—for example, Wasserstein-1 distance (OT) is the MAE version of bipartite matching loss in the 1D discrete setting. We also compared MAE, MSE, and Huber as the pairwise regression term and observed similar evaluation performance.

(2) Semi-supervised baselines
We compare **joint semi-supervised training** vs. **pretrain–finetune baseline** under several data combinations. Detailed results are shown in **Table A4** and **Reviewer pNG7's Response-6**.


<!-- ## Key Questions for Authors

> Can the authors quantify the noise level of ShiftDB-Lit more directly, for example by manually auditing a random subset of extracted examples? -->

#### Response-6 (Q1: noise level of ShiftDB-Lit)
See Response-3.


<!-- > How much of the gain comes from the sorting-based weak loss itself versus simply exposing the model to much broader chemical coverage? -->

#### Response-7 (Q2: gain from sorting-based weak loss)
See Response-1.


<!-- > Did the authors deduplicate near-identical molecules or control for overlap between literature-derived molecules and benchmark molecules? -->

#### Response-8 (Q3: deduplication and benchmark overlap)

We **deduplicate exact structures** and do not merge **near-identical** molecules with different structure.

We confirm that literature-derived entries **do not overlap** the benchmark data, and that ShiftDB-Lit has **no train–test overlap**.


<!-- > Could the authors compare against a Hungarian-loss implementation directly on a smaller subset to verify that the sorting surrogate is empirically equivalent in practice, not only theoretically? -->

#### Response-9 (Q4: numerical equivalence of losses)

We thank the reviewer for this suggestion. As a lightweight reproducibility check, we drew 10,000 random instances and compared the outputs of the two implementations. In every trial the two losses agreed within numerical tolerance. Code follows.

```python
import numpy as np
from scipy.optimize import linear_sum_assignment

rng = np.random.default_rng(0)
for _ in range(10000):
    n = int(rng.integers(2, 50))
    set1 = rng.uniform(0.0, 1.0, size=n)
    set2 = rng.uniform(0.0, 1.0, size=n)
    cost = np.abs(set1[:, None] - set2[None, :])
    r, c = linear_sum_assignment(cost)
    loss_hungarian = cost[r, c].sum()
    loss_sort = np.abs(np.sort(set1) - np.sort(set2)).sum()
    assert abs(loss_hungarian - loss_sort) < 1e-8
```


<!-- > For solvent modeling, what happens if the most common additional solvents are modeled separately rather than merged into “others”? -->

#### Response-10 (Q5: fine-grained solvent embedding)
See Response-4.


## Limitations

> The paper would be stronger with one extra section on data auditing: extraction quality, solvent label normalization, duplicates, and error examples. It would also help to separate the claims more clearly:
> - “semi-supervised objective is effective,”
> - “large-scale literature data improves coverage,”
> - “solvent conditioning adds value.”

> Right now these are somewhat intertwined. I would also like a clearer comparison to other possible permutation-invariant objectives and perhaps a discussion of when the sorting reduction fails if the loss assumptions are violated.

[Reference Response]


[TODO: response]

