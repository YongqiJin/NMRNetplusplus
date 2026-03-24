# Official Review of Submission2786

**Reviewer:** Reviewer pNG7  
**Date:** 12 Mar 2026, 23:53 (modified: 24 Mar 2026, 20:25)  
**Visibility:** Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer pNG7

---

## Summary

This paper introduces a semi-supervised framework for learning NMR chemical shifts from millions of unassigned spectra extracted from the literature. It frames chemical shift prediction from spectra as a permutation-invariant set supervision problem. The authors also curate ShiftDB-Lit, a large-scale dataset comprising millions of literature-extracted NMR spectra, and incorporate experimental solvent information into the learning process. This enables more accurate, solvent-aware, and multi-element chemical shift predictions.

---

## Strengths and Weaknesses

### Strengths

1. The paper is clearly written and easy to follow, and the open-source data and code are valuable for reproducibility.
2. Experimental NMR chemical shifts are inherently solvent-dependent, so incorporating a solvent prior is a good idea.

### Weaknesses

1. The ShiftDB-Lit dataset cannot be considered as a contribution of this paper, as it is simply a filtered version of the original dataset[1] using a systematic three-stage process.
[1] Nmrextractor: leveraging large language models to construct an experimental nmr database from open-source scientific publications
2. In Line 209, the authors formulate the weakly-supervised (molecule-level) loss as a bipartite matching loss and obtain the minimum by sorting the predicted and observed shifts and matching them in order. I am curious whether the sorting is done in ascending or descending order, and whether this choice would affect the final result.
3. In the experiments section, the authors should specify which datasets are used for supervised and weakly‑supervised training, along with their respective data ratios.
4. Table 2 is confusing. What does the superscript “3” indicate? Why are there so many blank entries? Moreover, the baseline setting is unfair. As the authors point out, the ~60% improvement is largely due to the baseline being evaluated under an OOD test.
5. I strongly suggest that the authors include more convincing baselines to demonstrate the effectiveness of the semi‑supervised training framework. For instance, first training with the weakly‑supervised molecule‑level loss and then fine‑tuning with the supervised atom‑level loss would help validate the carefully designed loss function. Additionally, an unsupervised baseline using a common masking strategy would also be a strong baseline.
6. I am confused about why 3D molecular conformations are generated in Line 130, as I could not find any experiments that actually make use of this 3D information.

---

## Scores

| Criterion     | Score | Label |
|---------------|-------|-------|
| Soundness     | 2     | fair  |
| Presentation  | 2     | fair  |
| Significance  | 2     | fair  |
| Originality   | 2     | fair  |

---

## Key Questions for Authors

See above

---

## Limitations

This paper does not discuss its limitations.

---

## Final Decisions

| Item | Value |
|------|--------|
| **Overall Recommendation** | **2: Reject** — Examples cited include technical flaws, weak evaluation, and inadequate reproducibility. |
| **Confidence** | **3** — You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that math/other details were not carefully checked. |
| **Compliance with LLM Reviewing Policy** | Affirmed |
| **Code of Conduct Acknowledgement** | Affirmed |
