# Response to Reviewer pNG7 (review4)

## Weaknesses

> The ShiftDB-Lit dataset cannot be considered as a contribution of this paper, as it is simply a filtered version of the original dataset[1] using a systematic three-stage process.
> [1] Nmrextractor: leveraging large language models to construct an experimental nmr database from open-source scientific publications

[Reference Response]
正文数据来源为 **NMRexp**（`wang2025nmrexp`，见 `main_context.tex` 约 69–71 行），而非 Reviewer 所列 NMRextractor 条目 [1] 的同一管线。ShiftDB-Lit 的贡献包括：在 **NMRexp** 上的**三阶段过滤**、与 **NMRShiftDB2 联合训练**可用的弱监督格式、**溶剂与异核**元数据及文中报告的基准数字。我们将更明确引用数据源与 [1] 的关系，避免与 “Nmrextractor-only 过滤” 混淆。

[TODO: response]

[TODO: Exp]
None.

> In Line 209, the authors formulate the weakly-supervised (molecule-level) loss as a bipartite matching loss and obtain the minimum by sorting the predicted and observed shifts and matching them in order. I am curious whether the sorting is done in ascending or descending order, and whether this choice would affect the final result.

[Reference Response]
当前 PDF 行号可能因排版变化；在 Methods 中已写明：将 \(\{s_i\}\) 与 \(\{\hat{s}_i\}\) 按**升序**排列后按序配对（约 126–132 行）。在定理条件满足时，升序匹配为最优；**降序等价于对其中一侧取反后的升序**，在单调损失下与「两侧同向排序」一致；若损失为 \(f(|x-y|)\) 且 \(f\) 单调凸，**排序方向不改变最优匹配**（与匈牙利最优解一致）。

[TODO: response]

[TODO: Exp]
None.

> In Lines 250–259, the authors categorize solvents into three groups: (1) CDCl₃ (89.1%), (2) DMSO‑d₆ (5.7%), and (3) other infrequent solvents, and encode them as learnable embeddings. How is this categorical information encoded, and what is the difference among the three category embeddings?

[Reference Response]
xxx

[TODO: response]

[TODO: Exp]
None.


> In the experiments section, the authors should specify which datasets are used for supervised and weakly‑supervised training, along with their respective data ratios.

[Reference Response]
消融表 Table~\ref{tab:ablation-dataset}（约 296 行起）列出 Exp.1–5：监督来自 **NMRShiftDB2**（\(L_{\mathrm{atom}}\)），弱监督来自 **ShiftDB-Lit 或 NMRShiftDB2**（\(L_{\mathrm{mol}}\)）；半监督主实验为 **NMRShiftDB2 + ShiftDB-Lit**。**batch 比例**由 \(B_1,B_2\) 与 \(\lambda\) 共同决定（约 146–151 行），Table~\ref{tab:para} 给出具体超参。修改稿将在 Implementation 首段用**一句话**汇总数据组合与有效采样比例。

[TODO: response]

[TODO: Exp]
None.（写作澄清；若需数字精确到每 step 比例，从训练配置抄入。）

> Table 2 is confusing. What does the superscript “3” indicate? Why are there so many blank entries? Moreover, the baseline setting is unfair. As the authors point out, the ~60% improvement is largely due to the baseline being evaluated under an OOD test.

[Reference Response]
（1）上标 “3” 应以**贵方 camera-ready PDF** 中 Table 2 的**表注/脚注**为准（常见为文献引用或训练设定说明）；rebuttal 中逐字引用表注含义即可；若为行内上标则对应该行脚注定义。（2）**空白格**：ShiftDB-Lit **无原子标注**故无 \(L_{\mathrm{atom}}\)；部分方法未在弱监督设定下评估故无 \(L_{\mathrm{mol}}\)。（3）**公平性**：同 Review 1/3，大降幅含 OOD→ID 因素；正文约 205–207 行已说明；修改稿将强化措辞。

[TODO: response]

[TODO: Exp]
核对 PDF 中 Table 2 脚注号与空白原因；必要时补 \(L_{\mathrm{mol}}\) 可算项（与 Review 2 同）。

> I strongly suggest that the authors include more convincing baselines to demonstrate the effectiveness of the semi‑supervised training framework. For instance, first training with the weakly‑supervised molecule‑level loss and then fine‑tuning with the supervised atom‑level loss would help validate the carefully designed loss function. Additionally, an unsupervised baseline using a common masking strategy would also be a strong baseline.

[Reference Response]
当前主文联合训练 \(L_{\mathrm{atom}}+\lambda L_{\mathrm{mol}}\)（约 139–151 行）。**两阶段**（先弱后强微调）是合理对照，可检验优化路径是否优于联合；**masking 自监督**需额外任务设计，与 NMR 位移回归目标不完全同构。我们可在 Discussion 中说明范围，并在资源允许时加**小规模**两阶段或简化自监督对照。

[TODO: response]

[TODO: Exp]
可选：weak-pretrain → supervised finetune；轻量 graph masking 预训练再微调（缩小数据与 epoch）。

> I am confused about why 3D molecular conformations are generated in Line 130, as I could not find any experiments that actually make use of this 3D information.

[Reference Response]
**NMRNet 骨干为 SE(3)-等变 Transformer，输入为原子 3D 坐标**（约 156–157 行）；预处理使用 **RDKit 嵌入 3D 构象**（约 77 行）。因此 3D **并非单独一节实验**，而是**所有表格中 NMRNet 系模型的默认输入**。若行号指向 Methods，我们将在修改稿交叉引用至 Architecture 小节，避免读者以为 3D 未使用。

[TODO: response]

[TODO: Exp]
None.

## Limitations

> This paper does not discuss its limitations.

[Reference Response]
将在修改稿增加独立 **Limitations**：文献噪声、构象/溶剂简化、随机划分与 scaffold、弱监督假设、排序损失适用条件等（与 Review 1/2 合并撰写，避免重复罗列）。

[TODO: response]

[TODO: Exp]
None.
