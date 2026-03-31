<!-- # Response to Reviewer pNG7 (review4)

## Weaknesses -->

Thank you for your time and thoughtful reviews! We address your comments point by point as follows.

**("W" refers to Weaknesses, "Q" refers to Key Questions)**

<!-- #### Response-1 (W1: contribution)

> The ShiftDB-Lit dataset cannot be considered as a contribution of this paper, as it is simply a filtered version of the original dataset[1] using a systematic three-stage process.
> [1] Nmrextractor: leveraging large language models to construct an experimental nmr database from open-source scientific publications -->

#### Response-1 (W1: dataset contribution & provenance)

**Section 3.1** states that our chemical-shift corpus is built from NMRexp, not from NMRextractor (the source cited in the comment). The upstream corpus is produced by a pipeline that combines PDF parsing, OCSR, and LLM-based extraction; it is relatively crude and not tailored to chemical-shift prediction. Our contribution on the data side is the task-oriented curation: spectrum-text parsing, multi-stage filtering and cleaning, and 3D conformer generation, yielding a higher-quality, benchmark-ready resource for this setting—rather than a trivial filter of an existing shift-prediction dataset.

We thank the reviewer for the pointer: NMRextractor is related work on literature-based NMR databases. We will discuss it in Related Work and include it in References.


<!-- > In Line 209, the authors formulate the weakly-supervised (molecule-level) loss as a bipartite matching loss and obtain the minimum by sorting the predicted and observed shifts and matching them in order. I am curious whether the sorting is done in ascending or descending order, and whether this choice would affect the final result. -->

#### Response-2 (W2: sorting direction)

We sort predicted and observed shifts in the **same** direction (both ascending or both descending) and pair them index-wise after sorting. Switching from ascending to descending reverses both sequences, so the sum of absolute differences—and hence the loss—is unchanged.

In addition, **Reviewer iTNC's Response-9** includes a short script that walks through the Hungarian and sorting-based computations. It helps clarify how the loss is implemented.


<!-- > In Lines 250–259, the authors categorize solvents into three groups: (1) CDCl₃ (89.1%), (2) DMSO‑d₆ (5.7%), and (3) other infrequent solvents, and encode them as learnable embeddings. How is this categorical information encoded, and what is the difference among the three category embeddings? -->

#### Response-3 (W3: solvent category embeddings)

The three coarse groups—CDCl₃, DMSO‑d₆, and a pooled “other” bucket—are defined in **Embedding Solvent Information** (Section 3.4, Methods) to handle the long-tailed solvent distribution: the two dominant solvents each get a category, while all remaining solvents share **one** learnable embedding as a practical approximation under data scarcity.

**Encoding.** Each example carries a discrete solvent category id in {1, 2, 3}. We look up a learnable vector e_solv with the same width as the backbone (`d_model`) from a standard embedding table with **three rows** (one per category); the same injection scheme applies to every category, as described in that subsection. In code: `nn.Embedding(3, encoder_embed_dim)`.

**Difference among the three embeddings.** They are three **independently learned** vectors of the same shape, trained end-to-end—not hand-crafted descriptors. The only structural difference is **which solvents map to which row**.



<!-- > In the experiments section, the authors should specify which datasets are used for supervised and weakly‑supervised training, along with their respective data ratios. -->

#### Response-4 (W4: supervised vs. weakly-supervised data and ratios)

**Datasets**. The Ablation subsection (**Section 4.5**) summarize which datasets are used for supervised and weakly‑supervised training; our strongest setting uses **NMRShiftDB2** for supervised training and **ShiftDB-Lit** for weakly supervised training.

**Batching (“ratios” in practice).** As in Methods, we use separate batch sizes **B₁** for supervised and **B₂** for weakly supervised losses, and we scale the weak term by **λ** (see lines 200–204). **Table 7 (training configuration)** reports **B₁=4**, **B₂=16**, and **λ=16**.

Thank the reviewer for the suggestion. We will add a one-sentence pointer in the experiments section so readers see the datasets, batch sizes, and λ together.


<!-- > Table 2 is confusing. What does the superscript “3” indicate? Why are there so many blank entries? Moreover, the baseline setting is unfair. As the authors point out, the ~60% improvement is largely due to the baseline being evaluated under an OOD test. -->

#### Response-5 (W5: Table 2)

- **Superscript “3”.** It denotes a **footnote** that lists the sources of the reported metrics for each method (see the footnote at the **bottom of the same page**).
- **Blank entries.** In the original table we prioritized NMRShiftDB2 as the primary benchmark; we have now completed the L_mol baseline columns under the same protocol in **Table A2 (Reviewer 6gyS's Response-1)**.
- **OOD test.** We agree the ShiftDB-Lit headline comparison is partly OOD vs. ID. **NMRShiftDB2** remains the **primary** benchmark; **ShiftDB-Lit** highlights how semi-supervised training **expands chemical-space coverage** beyond fully labeled regimes—closer to **deployment settings** where atom-level assignments are limited. 

Following Reviewer wmHV’s suggestion, we additionally perform a **structure-based split** for a fairer OOD evaluation, and report the detailed results in **Table A3 (Reviewer wmHV's Response-1)**.


<!-- > I strongly suggest that the authors include more convincing baselines to demonstrate the effectiveness of the semi‑supervised training framework. For instance, first training with the weakly‑supervised molecule‑level loss and then fine‑tuning with the supervised atom‑level loss would help validate the carefully designed loss function. Additionally, an unsupervised baseline using a common masking strategy would also be a strong baseline. -->

#### Response-6 (W6: training schedules & masking pretraining)

We thank the reviewer for the constructive suggestion. We report **(i)** staged training vs. joint semi-supervised learning and **(ii)** **Uni-Mol** initialization (3D masking pretraining on 209M conformers [1]) vs. training from scratch.

**(i) Staged vs. joint training.** No schedule wins every column, but **weakly‑supervised pretraining then supervised finetuning** is consistently weakest. **Joint** semi-supervised training remains simple, effective, and competitive, and is our default.

**(ii) Masking / self-supervised pretraining.** NMRNet already initializes from **Uni-Mol** [1], a 3D molecular encoder pretrained with atom- and coordinate-level masking on 209M conformers. The **w/ vs. w/o Uni-Mol** rows below show a **minor gain** with pretrained weights.


Table A4: Semi-supervised training baselines (MAE).

- Training schedule

| | | **¹H** | | | **¹³C** | | |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pretrain | Finetune | DB2 L_atom | DB2 L_mol | Lit L_mol | DB2 L_atom | DB2 L_mol | Lit L_mol |
| Semi-sup. (DB2 & Lit) | — | **0.1709** | **0.1492** | 0.0559 | 0.9270 | 0.7765 | 0.5060 |
| Weak-sup. (Lit) | Sup. (DB2) | 0.1744 | 0.1537 | 0.0734 | 0.9345 | 0.7905 | 0.6287 |
| Sup. (DB2) | Semi-sup. (DB2 & Lit) | 0.1725 | 0.1503 | **0.0548** | 0.9185 | **0.7675** | **0.4985** |
| Semi-sup. (DB2 & Lit) | Sup. (DB2) | 0.1710 | 0.1496 | 0.0583 | **0.9182** | 0.7779 | 0.5336 |

- Uni-Mol initialization

| | **¹H** | | | **¹³C** | | |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| | DB2 L_atom | DB2 L_mol | Lit L_mol | DB2 L_atom | DB2 L_mol | Lit L_mol |
| w/ Uni-Mol | 0.1709 | 0.1492 | 0.0559 | 0.9270 | 0.7765 | 0.5060 |
| w/o Uni-Mol | 0.1727 | 0.1504 | 0.0543 | 0.9319 | 0.7799 | 0.5091 |

Full RMSEs match the manuscript tables; we will add these results to the appendix in the revised version.

[1] Zhou et al., *Uni-Mol: A Universal 3D Molecular Representation Learning Framework*, ICLR 2023.



<!-- > I am confused about why 3D molecular conformations are generated in Line 130, as I could not find any experiments that actually make use of this 3D information. -->

#### Response-7 (W7: 3D conformations)

We generate 3D conformations because the backbone is an SE(3)-equivariant Transformer that **takes atomic coordinates as input**. We state this in Related Work (Lines 75–76) and in the model-architecture paragraph (Lines 212–214). We will add one explicit sentence to make it more clear.


## Limitations

> This paper does not discuss its limitations.

[TODO]
