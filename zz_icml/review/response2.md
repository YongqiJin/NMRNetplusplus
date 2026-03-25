# Response to Reviewer 6gyS (review2)

## Weaknesses

> No error bars are provided in any of the result tables; it is very hard to tell if the differences are significant or not.

[Reference Response]
当前主表报告的是单次（或默认 seed）下的点估计 MAE/RMSE。我们可在修改稿中为主结果补充**不确定性**：例如多次随机种子重训的均值±标准差，或对测试集 bootstrap 的置信区间（具体格式与表格空间以 camera-ready 为准）。审稿人 Note 也提到补上 error bars 与 Table 2 可改善评分印象。

[TODO: response]

[TODO: Exp]
多次 seed 重训或 bootstrap 区间（主表关键行）。

> Many elements of Table 2 are left unfilled, but the reviewer fails to see the reason why they cannot be computed. Any prediction method can be applied to the structures in the ShiftDB-Lit database, and the \(L_{\mathrm{mol}}\) loss can be computed on the prediction. (The reviewer may be missing something—in that case the authors should clarify—but they see no reason why the missing values cannot be computed in Table 2.)

[Reference Response]
正文说明：在 **ShiftDB-Lit** 上**无原子级标注**，因此 **\(L_{\mathrm{atom}}\)** 等需要原子-峰对应的目标无法计算（见 `main_context.tex` 约 201–202 行）。\(\,L_{\mathrm{mol}}\,\) 原则上可对任意在 ShiftDB-Lit 结构上产生预测集的模型计算，但表中**未列方法**需在相同弱监督设定下重训或复现其管线，并非简单“填表”即可与本文公平对比；**缺失格**应解释为「未在该设定下训练/评估」或「该方法不输出与实验设定一致的集合预测」。我们将在修改稿表注/脚注中逐格说明。

[TODO: response]

[TODO: Exp]
在可行范围内补全 Table 2 中**可公平计算**的 \(L_{\mathrm{mol}}\)（需为其他 baseline 在 ShiftDB-Lit 上训练或统一推断流程）；若篇幅有限，至少给出脚注与计划。

> **Note from reviewer:** Addressing the evaluation problem (adding error bars, providing all losses in Table 2 that can be computed) would automatically result in +1 in their overall score.

[Reference Response]
与 Weakness 1–2 及 Key Q1 的回应一致：补齐 **error bars** 与 **Table 2 可解释/可算项** 直接回应审稿人评分条件。

[TODO: response]

[TODO: Exp]
同主表 uncertainty 与 Table 2 补全。

> The idea of using a set-based supervision loss is not particularly novel.

[Reference Response]
集合预测与排序匹配思想在 ML 中确有先例；本文侧重点包括：（1）**NMR 无对应文献谱**这一科学场景下的形式化；（2）在常用回归损失下将指派**等价**为排序配对并用于**百万级**训练（理论在附录）；（3）与 NMRShiftDB2+ShiftDB-Lit 结合的消融（如仅弱监督崩溃、需标注锚定等）。修改稿中将在 Related Work 中更明确对比与定位。

[TODO: response]

[TODO: Exp]
None.

## Key Questions for Authors

> Please provide error bars for MAE and RMSE values, and provide the missing values in Table 2.

[Reference Response]
同 Weakness 1–2：error bars 通过多 seed 或 bootstrap；Table 2 在澄清「何种指标在何种数据上可定义」的前提下补全或脚注说明不可比原因。

[TODO: response]

[TODO: Exp]
同主表 uncertainty + Table 2 补全/脚注（与上合并）。

> The term “unsupervised” is used many times in the paper, but the literature data is not unsupervised as there is a label—a set label. As properly named elsewhere, this is a weakly supervised setting. To avoid confusion, change occurrences of “unsupervised” to “weakly supervised” in the text; “semi-supervised” can also be changed to “weakly supervised” (except for the title, which cannot be changed anymore).

[Reference Response]
同意审稿人：**文献侧为集合标签**，属弱监督/半监督范畴而非「完全无监督」。全文将系统替换误用的 “unsupervised” 为 “weakly supervised” 或 “unassigned spectra” 等更准确表述；**semi-supervised** 是否与 weakly supervised 完全统一，将在修改稿中保持术语表一致（标题不可改则正文与摘要统一说明）。

[TODO: response]

[TODO: Exp]
None.

> It seems the authors ignore multiplicity of \(^1\)H NMR peaks altogether. This is easy to use and valuable information to restrict possible permutations. Did they try using it?

[Reference Response]
当前管线从文献提取的是**化学位移集合/多重集**表示（`main_context.tex` 约 71 行），与标准 peak list 一致；**积分/多重性**若在原始文本中未稳定解析或未进入 NMRexp 字段，则未作为硬约束加入损失。我们未在正文报告 multiplicity 约束实验。若数据侧可可靠获得 multiplicity，可作为未来工作或 rebuttal 小规模试验。

[TODO: response]

[TODO: Exp]
可选：在子集上尝试用 multiplicity 缩小合法排列或作为辅助特征（依赖标注可用性）。

> On Figure 3, the model-collapse argument about the red line in the case of \(^1\)H seems fragile given that the \(^{13}\)C line does not show this behaviour. Nothing is mentioned about this in the text. Please elaborate on possible reasons.

[Reference Response]
正文约 311–314 行已讨论：在 NMRShiftDB2 的 \(L_{\mathrm{atom}}\) 上，\(^1\)H 与 \(^{13}\)C 均呈 U 形；并指出在 **ShiftDB-Lit 的 \(L_{\mathrm{mol}}\)** 指标上，\(^1\)H 随 \(\lambda\) 增大更易出现**分子级损失下降而原子对应退化**的现象。\(^{13}\)C 曲线形态差异可能与**谱复杂度、化学位移展宽、弱监督下匹配歧义程度**不同有关（\(^1\)H 溶剂与交换更敏感，已在溶剂节讨论 \(^1\)H 更易受溶剂影响）。我们将在修改稿中**显式对比两子图**并缩短与 collapse 段落的距离，避免读者觉得未解释。

[TODO: response]

[TODO: Exp]
None.（除非补充更细 \(\lambda\) 点，见 Review 3 W4。）

> How many conformers do you generate and use as input?

[Reference Response]
数据预处理使用 **RDKit** 生成分子 3D 构象作为几何输入（约 77 行）。实现上 `uninmr` 任务提供 `--conf-size`（默认 10）与 `--conformer-augmentation`（可选多构象增强）；**具体实验是否开启增强及使用几条构象**，将在修改稿的 Implementation/附录中按**实际运行配置**写清（请对照你们训练脚本或默认参数填写最终数字）。

[TODO: response]

[TODO: Exp]
None.（文档化即可；若不同核种设置不同需统一说明。）

> **Minor:** Using \(\hat{s}\) as ground truth and \(s\) as the prediction goes against traditional use, where a hat denotes the estimate; it would be more appropriate to switch the two notations.

[Reference Response]
当前稿中约定为：\(\{s_i\}\) 预测、\(\{\hat{s}_i\}\) 真值（约 99–100 行）。我们同意常见习惯为 \(\hat{s}\) 表示估计值；**修改稿将互换符号**并全文统一，避免与惯例冲突。

[TODO: response]

[TODO: Exp]
None.

## Limitations

> No explicit discussion of limitations is provided. One that comes to mind is the difficulty of guessing the dominant conformer(s) in the sample, as that can depend on the solvent for example; furthermore there can be an ensemble of conformers contributing to the signal.

[Reference Response]
将在修改稿增加 **Limitations** 段落：包括（1）实验谱对应**溶液中构象系综**而模型输入为离散 3D 结构；（2）溶剂与温度等未完全建模；（3）文献提取噪声与弱标签；（4）排序损失假设不适用时的局限。与 Review 1 Limitations、Review 4 合并避免重复。

[TODO: response]

[TODO: Exp]
None.
