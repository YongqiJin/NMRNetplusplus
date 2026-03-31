# Response to Reviewer pNG7 (review4)

## Weaknesses

> The ShiftDB-Lit dataset cannot be considered as a contribution of this paper, as it is simply a filtered version of the original dataset[1] using a systematic three-stage process.
> [1] Nmrextractor: leveraging large language models to construct an experimental nmr database from open-source scientific publications

[Response-1] [TODO]
Contribution.


> In Line 209, the authors formulate the weakly-supervised (molecule-level) loss as a bipartite matching loss and obtain the minimum by sorting the predicted and observed shifts and matching them in order. I am curious whether the sorting is done in ascending or descending order, and whether this choice would affect the final result.

[Response-2]
We sort the predicted and observed shifts in the same direction—either both ascending or both descending—and match them in order. Thus, the choice of ascending or descending does not change the result.

We further provide a small script, `zz_icml/review/code.py`, to verify the equivalence of the two approaches and to illustrate the point more clearly.


> In Lines 250–259, the authors categorize solvents into three groups: (1) CDCl₃ (89.1%), (2) DMSO‑d₆ (5.7%), and (3) other infrequent solvents, and encode them as learnable embeddings. How is this categorical information encoded, and what is the difference among the three category embeddings?

[Response-3] [TODO]



> In the experiments section, the authors should specify which datasets are used for supervised and weakly‑supervised training, along with their respective data ratios.

[Response-4]
In the Table 5 of Section 4.5 (Ablation Study), we specify the baseline (the best performance) used NMRShiftDB2 for supervised and ShiftDB-Lit for weakly-supervised training. Their respective data ratios are shown in Appendix D: batchsize 4 for supervised and batchsize 16 for weakly-supervised. We will emphasize this in the experiment section in the revised version.


> Table 2 is confusing. What does the superscript “3” indicate? Why are there so many blank entries? Moreover, the baseline setting is unfair. As the authors point out, the ~60% improvement is largely due to the baseline being evaluated under an OOD test.

[Response-5]
(1) The superscript "3" 是脚注，引用了数据的来源. 清楚的写在表格所在页的左下角.
(2) The blank entries are due to 我们认为nmrshiftdb2是主要的基准，因此没有将之前的方法在shiftdb-lit上评估. 为了更完整的评估，我们补充了这些方法在shiftdb-lit上的结果.见
(3) OOD


> I strongly suggest that the authors include more convincing baselines to demonstrate the effectiveness of the semi‑supervised training framework. For instance, first training with the weakly‑supervised molecule‑level loss and then fine‑tuning with the supervised atom‑level loss would help validate the carefully designed loss function. Additionally, an unsupervised baseline using a common masking strategy would also be a strong baseline.

[Response-6]
Thank the reviewer for the suggestion. We add a new table [Table-D1] to show the performance of the weakly-supervised training and the supervised training.

[Table-D1]


> I am confused about why 3D molecular conformations are generated in Line 130, as I could not find any experiments that actually make use of this 3D information.

[Response-7]
NMRNet "uses an SE(3)-equivariant Transformer, taking molecular spatial coordinates as input". We mention that in the Line 075-076 (Related Work) and
Line 212-214 (Model Architecture). To make it more clear, we will emphasize this in the revised version.


## Limitations

> This paper does not discuss its limitations.

[TODO]
