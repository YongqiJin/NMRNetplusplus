# Response to Reviewer wmHV (review3)

## Weaknesses (W1–W6)

> **W1 (Major Soundness).** The ShiftDB-Lit test set evaluation conflates two effects. The NMRNet baseline is trained only on NMRShiftDB2, making ShiftDB-Lit out-of-distribution for it, while the semi-supervised model trains on ShiftDB-Lit, making it in-distribution. The reported reductions ¹H MAE from 0.1395 to 0.0559 (↓59.9%) and ¹³C MAE from 1.2591 to 0.5060 (↓59.8%) therefore reflect both the value of semi-supervised learning and simple distribution coverage. The authors acknowledge this (lines 283–293) but do not attempt to disentangle the two contributions for instance, by evaluating on a held-out scaffold split that is OOD for both models. This is fixable: a scaffold-based partition of ShiftDB-Lit that excludes training scaffolds from both models would isolate the semi-supervised learning gain.

[Reference Response]
与 `main_context.tex` 约 205–207 行一致：已承认 baseline 与半监督模型在 ShiftDB-Lit 上的 **OOD/ID 差异**。审稿人建议的 **scaffold（或结构骨架）held-out** 可使两种训练方式在**同一测试分子集合**上比较，从而分离「见没见过 ShiftDB-Lit 训练域」与「半监督目标」。我们同意这是更严格的解读，将在修改稿或 rebuttal 中报告**随机划分 vs scaffold 划分**结果（若时间允许完成）。

[TODO: response]

[TODO: Exp]
在 ShiftDB-Lit 上构造 scaffold split 测试集：仅 NMRShiftDB2 训练的模型与半监督模型**均在未见骨架**上评估；报告 MAE/RMSE 并与 Table 主结果对照。

> **W2 (Major Soundness).** ShiftDB-Lit uses a random 4:1 train/test split (Section 4.1) rather than a scaffold split. For a dataset of ~1.6M molecules, random splitting almost certainly places structurally similar molecules on both sides of the partition, inflating reported metrics. The NMRShiftDB2 results use a pre-defined benchmark split and are less affected, but the ShiftDB-Lit numbers which show the most dramatic gains (59.9% and 59.8% MAE reductions) and carry the paper's strongest narrative claims are the most vulnerable to this concern. This is also fixable by re-evaluating with a scaffold split.

[Reference Response]
正文约 187 行写明 ShiftDB-Lit 为 **4:1 随机划分**；NMRShiftDB2 沿用**既有 benchmark 划分**。随机划分确实可能造成结构相似泄漏；**scaffold split** 会更保守。我们将在修改稿中报告 scaffold 结果或至少讨论该局限与随机划分数字的上界性质。

[TODO: response]

[TODO: Exp]
与 W1 同一套 scaffold 实验；可附「random vs scaffold」对照表。

> **W3 (Major Soundness).** The solvent embedding is limited to three categories: CDCl₃ (89.1% of data, 162,509 test molecules), DMSO-d₆ (5.7%, 11,623 test molecules), and "others" (5,553 test molecules for ¹H). Table 3 shows the catch-all "others" category retains the highest MAE (0.0996 for ¹H, 0.8684 for ¹³C with solvent incorporation), and its improvement over the solvent-agnostic baseline is modest (↓9.5% for ¹H, ↓9.0% for ¹³C) compared to the gains for DMSO-d₆ (↓46.8% for ¹H, ↓17.8% for ¹³C). There is no analysis of (a) whether the "others" embedding does meaningful work beyond what the model learns without it, or (b) whether finer solvent granularity (e.g., 5–10 categories) would improve performance. Given that ShiftDB-Lit presumably contains solvent identity metadata beyond these three bins, the coarse grouping appears to be an underexplored design choice rather than a hard constraint. A targeted ablation comparing 3-bin vs. finer-grained solvent embeddings would address this.

[Reference Response]
三桶与表内数字来自正文与 Table~\ref{tab:solvent-all} 设定（约 165–166、252–256 行）。**“others” 误差更高**与长尾溶剂化学异质性一致；相对无溶剂基线的提升在文中已有方向性讨论。修改稿将补充 **“others” 内溶剂频率分布**（若空间允许）及是否相对无溶剂仍有净收益的讨论；**更细粒度 embedding** 可作为消融。

[TODO: response]

[TODO: Exp]
统计 test 集 “others” 中溶剂标签直方图；可选 5–10 类细粒度 vs 三桶 MAE 对比。

> **W4 (Minor Soundness).** The \(\lambda\) sweep (Figure 3) uses powers of two from \(2^0\) to \(2^7\); the optimum is \(\lambda=16\) (\(2^4\)). On NMRShiftDB2 (\(L_{\mathrm{atom}}\)), \(^1\)H MAE is ~0.170 at \(\lambda=16\) vs. ~0.190 at \(\lambda=128\); \(^{13}\)C MAE ~0.93 vs. ~1.15 over the same range. A finer grid near the optimum or a sensitivity curve would show how sharp the basin is. Easy to add.

[Reference Response]
同意：在 \(\lambda=16\) 附近加密网格或补一条局部敏感性曲线可展示最优域**平坦度**。属低成本补充，可在修改稿附录或主文一句完成。

[TODO: response]

[TODO: Exp]
在 \(\lambda \in \{8,12,16,20,24,32\}\)（或类似）上补点（若算力允许仅验证 \(L_{\mathrm{atom}}\)）。

> **W5 (Minor Presentation).** The heteroatom results (Table 4) report MAE values of 2.2809 ppm (¹⁹F), 1.3099 ppm (³¹P), 0.8287 ppm (¹¹B), and 1.9186 ppm (²⁹Si), with R² values of 0.7216, 0.9634, 0.9406, and 0.8901, respectively. However, these are presented without any baselines or comparison methods, making it difficult to contextualize whether these numbers represent strong or weak performance. Even a simple DFT or HOSE comparison would anchor them. The ¹⁹F R² of 0.7216 in particular seems relatively low and warrants discussion.

[Reference Response]
Table~\ref{tab:hete}（约 271–284 行）为 ShiftDB-Lit 上异核**监督学习**结果；未与 DFT/经验规则系统对比主要是篇幅与成本。修改稿将增加**讨论**：\(^{19}\)F **化学位移范围宽**（见 Table 6）、数据噪声与偶极展宽等可致 \(R^2\) 相对 \(^{31}\)P、\(^{11}\)B 更低；并可选**小样本** DFT 或文献 HOSE 对比作为参照（若 rebuttal 时间允许）。

[TODO: response]

[TODO: Exp]
可选：对少量分子 DFT 化学位移或 HOSE 估计与模型对比的小表；或仅讨论 + 引用范围。

> **W6 (Minor Presentation).** Wall-clock training time is not reported. Table 7 shows 10 epochs for semi-supervised vs. 50 for supervised, with batch sizes of 4 (labeled) + 16 (unlabeled) for semi-supervised vs. 8 (labeled only) for supervised, and learning rates of 4e-4 vs. 1e-4 respectively. Since the weakly-supervised data contains 898,422 ¹H and 704,373 ¹³C entries compared to NMRShiftDB2's 12,800 ¹H and 26,859 ¹³C entries (~26-70× larger), the actual computational cost is difficult to assess from epoch counts alone. This matters for practitioners deciding whether to adopt the method.

[Reference Response]
实验在 **NVIDIA RTX 4090** 上运行（约 189 行）。**每 epoch 墙钟时间**与总训练时间可从现有日志提取；半监督虽 epoch 少但每步需处理更大弱监督 batch。我们将在修改稿 Table~\ref{tab:para} 附近或附录补充 **hours/epoch** 与**总时长**（以你们 `finetune.log` 等实测为准填写数字）。

[TODO: response]

[TODO: Exp]
None.（从已有运行记录汇总 wall-clock；若无记录则补跑一次计时。）

## Key Questions For Authors

> Can you evaluate on a scaffold-split held-out set from ShiftDB-Lit that is OOD for both the baseline and semi-supervised model? This would directly address W1 and W2 by isolating the semi-supervised learning gain from distribution coverage.

[Reference Response]
同 W1/W2：这是同一问题的操作化表述。理想实验为 scaffold 测试集上**两模型均不在训练集中含相同骨架**（或按 Bemis–Murcko 等定义），再比较相对提升。

[TODO: response]

[TODO: Exp]
同 Weakness W1/W2 scaffold 实验。

> What is the solvent distribution within the “others” category (5,553 \(^1\)H test molecules, 5,485 \(^{13}\)C test molecules), and have you tried finer granularity? The ↓9.5% (\(^1\)H) and ↓9.0% (\(^{13}\)C) improvements for “others” vs. ↓46.8% and ↓17.8% for DMSO-d\(_6\) suggest the single embedding may be underperforming. (Addresses W3.)

[Reference Response]
我们将在修改稿中报告 **others 内**各溶剂（或标准化名称）的频数/占比；并讨论单 embedding 对异质溶剂的局限。细粒度 embedding 消融见 W3。

[TODO: response]

[TODO: Exp]
同 W3：分布统计 + 可选细粒度 ablation。

> What are the wall-clock training times for the supervised (50 epochs, batch size 8) vs. semi-supervised (10 epochs, batch sizes 4+16) settings on the reported NVIDIA RTX 4090? (Addresses W6.)

[Reference Response]
同 W6：从 4090 上实际训练日志汇总两种设定的 **总墙钟时间** 与 **每 epoch 时间**（注明数据加载是否含 I/O 瓶颈）。

[TODO: response]

[TODO: Exp]
None.（汇总日志。）

> For \(^{19}\)F, the \(R^2\) of 0.7216 is notably lower than for \(^{31}\)P (0.9634) and \(^{11}\)B (0.9406). Can you comment on whether this reflects intrinsic difficulty, data noise, or the broader chemical shift range (~300 to 300 ppm per Table 6)?

[Reference Response]
三者可能同时存在：\(^{19}\)F **标度范围大**（Table 6）会放大残差方差、压低 \(R^2\)；数据侧提取噪声与偶极展宽；以及氟环境电子效应多样导致**内在难拟合**。修改稿将用一段文字展开，并与 Table 6 范围一致引用。

[TODO: response]

[TODO: Exp]
None.（以讨论为主；可选补充 DFT 小样本。）
