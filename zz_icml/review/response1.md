# Response to Reviewer iTNC (review1)

## Weaknesses / concerns

> My main concern is evaluation fairness and interpretation. The paper emphasizes large gains on ShiftDB-Lit, but the baseline is trained only on NMRShiftDB2 while the proposed model leverages ShiftDB-Lit during training, so the comparison is partly OOD-vs-ID rather than purely method-vs-method. The paper does acknowledge this, but the framing should be more careful.

[Reference Response]
正文已写明：NMRNet baseline 仅在 NMRShiftDB2 上训练，因而在 ShiftDB-Lit 上为 OOD；半监督模型训练时使用了 ShiftDB-Lit，测试为 ID（见 `main_context.tex` 约 205–207 行）。这与「方法本身」和「分布覆盖」混杂有关。与之相对，**NMRShiftDB2 上的对比**（相对先前监督方法，\(^1\)H / \(^{13}\)C MAE 分别约降 13.4% / 19.6%）是在固定 benchmark 划分下的**同设定**改进，更适合作为「方法相对 SOTA」的主结论。我们将在修改稿中把两类结果分开展示（主文/图注措辞），避免把 ShiftDB-Lit 上的大降幅单独说成纯方法收益。

[TODO: response]

[TODO: Exp]
None
<!-- Scaffold（或结构 held-out）在 ShiftDB-Lit 上重评，使仅 NMRShiftDB2 训练的 baseline 与半监督模型对**同一**测试域可比（与 Review 3 重叠时可合并一次实验）。 -->

> A second concern is novelty level. The application is valuable, and the sorting reduction is elegant, but the overall method is still a relatively direct semi-supervised extension of an existing backbone rather than a fundamentally new model family. The strongest novelty is really the formulation plus the scale of the data resource.

[Reference Response]
与稿件表述一致：骨干为既有 NMRNet（SE(3)-等变），贡献侧重在（1）无原子对应文献谱的**置换不变集合监督**与排序损失等价形式；（2）**大规模** ShiftDB-Lit 与溶剂条件；（3）弱监督与标注数据结合的实证与消融（如 Exp.1–5）。修改稿中将在 Introduction/Contributions 中更明确区分「新架构」与「新学习设定 + 数据与资源」，避免过度声称全新模型族。

[TODO: response]

[TODO: Exp]
None.

> A third concern is data quality / noise robustness. The literature-extracted dataset is large, but inevitably noisy. The paper describes filtering procedures, which is good, but the main text does not quantify error rates from extraction, OCSR, parsing, solvent normalization, or duplicate handling. Since the paper’s central claim relies on learning from noisy literature data, this deserves more explicit auditing.

[Reference Response]
正文已描述**三阶段过滤**（分子有效性、NMR 有效性、一致性检查）及附录 `Appendix~\ref{app:data}` 指向（见 `main_context.tex` 约 73–77、82 行）。审稿人指出的**各环节错误率/去重比例等定量审计**目前主文未系统给出——我们承认这是加强点，将在修改稿中补充摘要性数字或附录表（来源：人工抽检子集、规则统计或与原始 NMRexp 条目对照等，以实际可复现为准）。

[TODO: response]

[TODO: Exp]
随机子集人工审计 + 各环节可量化指标（与 Review 1 Key Q1 及 Limitations 一并规划）。

> The solvent modeling is promising but still fairly coarse. Solvents are grouped into three categories, with “others” collapsed into one embedding. That is understandable for data imbalance reasons, but it limits chemical interpretability and may hide meaningful solvent-specific behavior.

[Reference Response]
正文已说明三分类动机：\ce{CDCl3} 与 \ce{DMSO-d6} 为主流，其余合并为 “others” 以应对**长尾与数据稀疏**（约 165–166 行）。我们同意细粒度溶剂可能带来更可解释或更优的嵌入；可在修改稿中补充 “others” 内分布描述，并视 rebuttal 篇幅讨论是否增加**细粒度溶剂 embedding 消融**。

[TODO: response]

[TODO: Exp]
细粒度溶剂：例如将 “others” 中高频溶剂单独设类或 5–10 类与三桶对比（与 Review 1 Q5 联动）。

> Finally, the paper would benefit from stronger baselines in the weak-supervision setting. Most comparisons are against older supervised predictors or the NMRNet baseline. There is less discussion of alternative set-prediction / permutation-invariant training objectives or semi-supervised baselines beyond the chosen formulation.

[Reference Response]
Table~\ref{tab:methods}（整体对比）已包含 HOSE、GCN、FCG、SGNN、GT-NMR、NMRNet 等**监督**基线；**弱监督设定**下替代目标（如匈牙利真匹配每步、两阶段预训练再微调、masking 自监督等）讨论与实验在篇幅上有限。我们可在修改稿中增加 Related Work/Discussion 中对替代集合目标的引用与讨论；若 rebuttal 允许，补充**小规模**对照实验以增强说服力。

[TODO: response]

[TODO: Exp]
None

## Key Questions for Authors

> Can the authors quantify the noise level of ShiftDB-Lit more directly, for example by manually auditing a random subset of extracted examples?

[Reference Response]
当前主文未给出人工抽检的定量结论；我们将在修改稿或附录中报告**随机子集**人工核对比例（例如结构解析错误、峰表异常、溶剂标注一致性等），并与自动过滤规则统计一并呈现（具体指标以实际完成为准）。

[TODO: response]

[TODO: Exp]
None

> How much of the gain comes from the sorting-based weak loss itself versus simply exposing the model to much broader chemical coverage?

[Reference Response]
消融表（Table~\ref{tab:ablation-dataset}，约 296–304 行）表明：仅在 NMRShiftDB2 上加 \(L_{\mathrm{mol}}\)（Exp.4）相对纯监督（Exp.1）**不**提升；**NMRShiftDB2 监督 + ShiftDB-Lit 弱监督**（Exp.5）带来提升，说明增益与**文献大规模弱标签数据**强相关，而非同一标注集上“排序损失形式”单独带来的收益。严格分离「仅扩充分子覆盖但不弱监督」需额外对照实验（若审稿人坚持）。

[TODO: response]

[TODO: Exp]
Unimol 预训练

> Did the authors deduplicate near-identical molecules or control for overlap between literature-derived molecules and benchmark molecules?

[Reference Response]
正文对**去重与 train/test 重叠**的定量表述需在修改稿中按实际数据处理管道补全（若当前流水线含基于 SMILES/InChI 或指纹的去重，将明确写出；与 NMRShiftDB2 分子的重叠可报告交集规模或 scaffold 重叠率）。此处以你们代码/日志为准在终稿中写死数字。

[TODO: response]

[TODO: Exp]
若主文尚无：统计 ShiftDB-Lit 与 NMRShiftDB2 的分子/骨架重叠比例；文献侧去重规则说明。

> Could the authors compare against a Hungarian-loss implementation directly on a smaller subset to verify that the sorting surrogate is empirically equivalent in practice, not only theoretically?

[Reference Response]
理论上证实在 MAE/MSE/Huber 等满足 \(l(x,y)=f(|x-y|)\) 且 \(f\) 单调凸时，最优匹配等价于**两侧排序后按序配对**（正文约 120–136 行，附录证明）；匈牙利算法在同一损失下应给出**相同**最优指派。实证上可在小批量或子集上核对 Hungarian 与排序的 \(L_{\mathrm{mol}}\) 是否一致（数值应在浮点误差内）。

[TODO: response]

[TODO: Exp]
子集上 Hungarian vs 排序损失数值一致性验证（或短程训练曲线对比）。

> For solvent modeling, what happens if the most common additional solvents are modeled separately rather than merged into “others”?

[Reference Response]
当前实现为三桶嵌入以平衡数据量（约 165–166 行）。将 “others” 中高频溶剂单独设 embedding 可能提升该子域表现；需用同一划分做 **ablation** 报告 MAE/RMSE（尤其 “others” 子集）。

[TODO: response]

[TODO: Exp]
多类溶剂 embedding 或 5–10 溶剂细分 vs 三桶 baseline。

## Limitations

> The paper would be stronger with one extra section on data auditing: extraction quality, solvent label normalization, duplicates, and error examples. It would also help to separate the claims more clearly:
> - “semi-supervised objective is effective,”
> - “large-scale literature data improves coverage,”
> - “solvent conditioning adds value.”

> Right now these are somewhat intertwined. I would also like a clearer comparison to other possible permutation-invariant objectives and perhaps a discussion of when the sorting reduction fails if the loss assumptions are violated.

[Reference Response]
我们计划在修改稿中：（1）增加**数据审计**相关段落或附录（与噪声/去重/溶剂归一化一致）；（2）用小节或 bullet **拆分三条 claim**，并与 NMRShiftDB2 结果、ShiftDB-Lit 结果、溶剂与 cross-solvent 实验分别对应；（3）在 Discussion 中补充：当损失**不**写成 \(f(|x-y|)\) 或 \(f\) 不满足单调凸时，排序不再保证等价于最优匹配，需回退一般指派或匈牙利/网络流等形式。

[TODO: response]

[TODO: Exp]
数据审计与 claim 拆分以写作与统计为主；若需新实验，与上文「人工子集审计」「溶剂消融」重叠则合并一次完成。
