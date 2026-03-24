# Official Review of Submission2786

**Reviewer:** Reviewer iTNC  
**Date:** 16 Mar 2026, 07:57 (modified: 24 Mar 2026, 20:25)  
**Visibility:** Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer iTNC

---

## Summary

The authors claim to address an important concept: overcoming the data bottleneck in NMR chemical shift prediction by combining a small atom-assigned dataset with millions of literature-extracted but unassigned spectra. This paper attempts to address the area of weakly/semi-supervised learning for scientific data where exact labels are expensive but weak structural signals are abundant. The core idea is to formulate unassigned NMR supervision as a permutation-invariant set prediction problem, then show that under common regression losses the matching objective reduces to a sorting-based loss, making large-scale training practical. The paper also introduces ShiftDB-Lit, a large literature-derived dataset with solvent metadata, and studies solvent-aware conditioning as well as heteroatom shift prediction. Empirically, the method improves over the NMRNet baseline on both NMRShiftDB2 and ShiftDB-Lit, with especially large gains on the literature-scale benchmark.

---

## Strengths and Weaknesses

### Strengths

1. The problem is important and well motivated. Existing ML methods depend on scarce atom-level assignments, while literature contains much larger amounts of unassigned spectra. Framing this as weak supervision is natural and potentially impactful beyond NMR.
2. The technical formulation is clean. The paper defines a bipartite matching objective over predicted and observed shifts, then argues that for losses of the form \(l(x, y) = f(|x - y|)\) with monotone convex \(f\), the optimum is achieved by sorting both sets and matching in order. This is a nice simplification that converts a combinatorial assignment problem into a deterministic training loss.
3. The dataset contribution is substantial. ShiftDB-Lit is much larger than NMRShiftDB2 and includes solvent information plus several heteroatoms. That makes the work useful not only methodologically but also as infrastructure for future research.
4. The experimental gains are strong. On NMRShiftDB2, semi-supervised training improves the NMRNet baseline for both ^1H and ^13C; on ShiftDB-Lit, the gains are much larger. The solvent-conditioning experiments are also interesting, especially the strong benefit on underrepresented solvents such as DMSO-d6.
5. I also appreciate the ablation in Table 5 showing an important negative result: weak supervision alone collapses, but works well when anchored by labeled data. That gives the paper a more credible and nuanced story than simply “more data helps.”

### Weaknesses / concerns

1. My main concern is evaluation fairness and interpretation. The paper emphasizes large gains on ShiftDB-Lit, but the baseline is trained only on NMRShiftDB2 while the proposed model leverages ShiftDB-Lit during training, so the comparison is partly OOD-vs-ID rather than purely method-vs-method. The paper does acknowledge this, but the framing should be more careful.
2. A second concern is novelty level. The application is valuable, and the sorting reduction is elegant, but the overall method is still a relatively direct semi-supervised extension of an existing backbone rather than a fundamentally new model family. The strongest novelty is really the formulation plus the scale of the data resource.
3. A third concern is data quality / noise robustness. The literature-extracted dataset is large, but inevitably noisy. The paper describes filtering procedures, which is good, but the main text does not quantify error rates from extraction, OCSR, parsing, solvent normalization, or duplicate handling. Since the paper’s central claim relies on learning from noisy literature data, this deserves more explicit auditing.
4. The solvent modeling is promising but still fairly coarse. Solvents are grouped into three categories, with “others” collapsed into one embedding. That is understandable for data imbalance reasons, but it limits chemical interpretability and may hide meaningful solvent-specific behavior.
5. Finally, the paper would benefit from stronger baselines in the weak-supervision setting. Most comparisons are against older supervised predictors or the NMRNet baseline. There is less discussion of alternative set-prediction / permutation-invariant training objectives or semi-supervised baselines beyond the chosen formulation.

---

## Scores

| Criterion     | Score | Label |
|---------------|-------|-------|
| Soundness     | 3     | good  |
| Presentation  | 3     | good  |
| Significance  | 3     | good  |
| Originality   | 3     | good  |

---

## Key Questions for Authors

1. Can the authors quantify the noise level of ShiftDB-Lit more directly, for example by manually auditing a random subset of extracted examples?
2. How much of the gain comes from the sorting-based weak loss itself versus simply exposing the model to much broader chemical coverage?
3. Did the authors deduplicate near-identical molecules or control for overlap between literature-derived molecules and benchmark molecules?
4. Could the authors compare against a Hungarian-loss implementation directly on a smaller subset to verify that the sorting surrogate is empirically equivalent in practice, not only theoretically?
5. For solvent modeling, what happens if the most common additional solvents are modeled separately rather than merged into “others”?

---

## Limitations

The paper would be stronger with one extra section on data auditing: extraction quality, solvent label normalization, duplicates, and error examples. It would also help to separate the claims more clearly:

- “semi-supervised objective is effective,”
- “large-scale literature data improves coverage,”
- “solvent conditioning adds value.”

Right now these are somewhat intertwined. I would also like a clearer comparison to other possible permutation-invariant objectives and perhaps a discussion of when the sorting reduction fails if the loss assumptions are violated.

---

## Final Decisions

| Item | Value |
|------|--------|
| **Overall Recommendation** | **4: Weak accept** — Technically solid paper that advances at least one sub-area of AI, with a contribution that others are likely to build on, but with some weaknesses that limit its impact (e.g., limited evaluation). Please use sparingly. |
| **Confidence** | **4** — You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. |
| **Compliance with LLM Reviewing Policy** | Affirmed |
| **Code of Conduct Acknowledgement** | Affirmed |
