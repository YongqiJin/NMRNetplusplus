# ICML 2026 rebuttal — 审稿待办（按 Review 原文 Weakness / Key Questions 映射）

说明：**重要性**、**容易程度**为建议档位（高/中/低；易/中/难）；**已解决**留空供你填写。

---

## 一、论文层面要解决的问题（写作 / 澄清 / 叙事 / 结构）

| # | 问题（简要） | 重要性 | 容易程度 | 已解决 | 对应 Review |
|---|-------------|--------|----------|--------|-------------|
| 1 | **公平比较与 claims 拆分**：明确区分「半监督目标本身」「文献数据带来的分布/覆盖」「溶剂条件」三类贡献；弱化或量化「~60% 提升」中 OOD baseline vs ID 训练混杂（正文/图注措辞一致） | 高 | 中 |  | Review 1: W1；Review 3: W1；Review 4: W4 |
| 2 | **ShiftDB-Lit 数据质量**：补充提取/OCSR/解析/溶剂归一化/去重等环节的**定量**说明；最好加「随机子集人工审计」小节或附录 | 高 | 中 |  | Review 1: W3；Review 1: Limitations |
| 3 | **Table 2**：补全可算的缺失格；解释空白与上标含义；必要时脚注说明与 baseline 的可比性 | 高 | 中 |  | Review 2: W2；Review 4: W3, W4 |
| 4 | **全文术语**：将误用的「unsupervised」改为「weakly supervised」（标题除外）；「semi-supervised」是否统一视你定义而定 | 中 | 易 |  | Review 2: Q2 |
| 5 | **符号习惯**：预测与真值符号（hat 戴在预测上） | 低 | 易 |  | Review 2: Minor |
| 6 | **排序损失**：说明 predicted/observed 按**升序还是降序**配对、是否在常见损失下等价（与附录一致） | 低 | 易 |  | Review 4: W2 |
| 7 | **实验设置可读性**：各阶段 supervised / weak 所用数据集与**样本比例**写清楚 | 中 | 易 |  | Review 4: W3 |
| 8 | **数据集贡献表述**：回应「仅为 NMRextractor 过滤版」— 说明过滤规则、规模、元数据与再现实验价值 | 中 | 中 |  | Review 4: W1 |
| 9 | **3D 构象**：说明 NMRNet 输入是否用 3D、本文是否仅用标量/不变特征；与实验对应 | 中 | 易 |  | Review 4: W6 |
| 10 | **Figure 3（λ 曲线）**：解释为何 ^1H 呈现「collapse」叙事而 ^13C 曲线不同；必要时补一句讨论 | 中 | 易 |  | Review 2: Q4；Review 3: W4 |
| 11 | **异核结果**：^19F 低 R² 等原因（难度、噪声、化学位移范围）；与 Table 6 一致 | 中 | 易 |  | Review 3: Q4；Review 3: W5 |
| 12 | **Limitations**：构象/溶剂依赖、弱标签噪声、排序假设失效情形等独立段落 | 中 | 易 |  | Review 2: Limitations；Review 4: Limitations；Review 1: Limitations（部分） |
| 13 | **新颖性表述**：强调「形式化 + 数据规模 + 任务」而非全新 backbone；与 Review 1 W2 对齐 | 中 | 易 |  | Review 1: W2；Review 2: W3 |

---

## 二、需要做的实验 / 计算（含「可只做小规模验证」的）

| # | 实验（简要） | 重要性 | 容易程度 | 已解决 | 对应 Review |
|---|-------------|--------|----------|--------|-------------|
| A | **Error bars**：对主表 MAE/RMSE 报告 seed 方差或 bootstrap 区间（Reviewer 2 明确：补上可 +1 分） | 高 | 中 |  | Review 2: W1；Review 2: Q1 |
| B | **Scaffold（或结构 held-out）划分**：在 ShiftDB-Lit 上重评，使 baseline 与半监督模型对同一 OOD 测试集可比；报告与 random split 对照 | 高 | 难 |  | Review 3: W1, W2；Review 3: Q1 |
| C | **分解增益**：弱监督损失本身 vs 仅扩大化学空间覆盖（例如仅额外结构/谱 coverage 的对照或分析） | 高 | 难 |  | Review 1: Q2 |
| D | **重叠/去重**：文献分子与 benchmark 分子的近重复或 scaffold 重叠统计与控制 | 高 | 中 |  | Review 1: Q3 |
| E | **Hungarian 真匹配 vs 排序损失**：在子集上对比二者训练/验证曲线，验证排序 surrogate 实证等价性 | 中 | 中 |  | Review 1: Q4 |
| F | **更强半监督 / 集合学习 baseline**：如 weak-pretrain → atom-supervised finetune；或常见 masking/自监督基线（规模可缩小） | 高 | 难 |  | Review 1: W5；Review 4: W5 |
| G | **溶剂 embedding**：「others」拆出若干常见溶剂单独 embedding 或 5–10 类细粒度 vs 三桶 ablation | 中 | 中 |  | Review 1: W4, Q5；Review 3: W3；Review 3: Q2 |
| H | **λ 敏感性**：在最优附近更细网格或曲线，说明盆地宽窄 | 低 | 易 |  | Review 3: W4 |
| I | **^1H multiplicity**：是否作为排列约束或特征试过（哪怕小规模） | 中 | 中 |  | Review 2: Q3 |
| J | **异核 baseline**：DFT、经验规则或 HOSE 类参考（至少讨论或一个小表） | 中 | 中 |  | Review 3: W5 |
| K | **Wall-clock**：监督 50 epoch vs 半监督 10 epoch 等在相同硬件上的总训练时间 | 中 | 易 |  | Review 3: W6；Review 3: Q3 |
| L | **构象数**：明确生成/采样构象数量（方法节 + 附录） | 低 | 易 |  | Review 2: Q5 |

---

## 三、审稿人整体立场（便于排优先级）

| Review | 总体建议 | 对你最有利的可操作点 |
|----------|----------|----------------------|
| Review 1 | Weak accept | 数据审计、公平叙事、额外 baseline |
| Review 2 | Weak reject | **error bars + Table 2** 明确可抬分 |
| Review 3 | Accept | scaffold 拆分、溶剂细粒度、时间/异核 |
| Review 4 | Reject | Table 2/公平比较、weak→finetune/masking baseline、3D 说明 |

---

*生成自 `zz_icml/review/review1.md`–`review4.md`，Weakness 编号以各文件内小节为准。*
