# Official Review of Submission2786

**Reviewer:** Reviewer 6gyS  
**Date:** 13 Mar 2026, 16:49 (modified: 24 Mar 2026, 20:25)  
**Visibility:** Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer 6gyS

---

## Summary

The authors propose a weakly supervised framework to learn an NMR chemical shift predictor from structure using literature-extracted weakly labeled NMR spectra and a small amount of strongly labeled (atom-level assigned) NMR spectra from NMRShiftDB2. Another contribution is to provide a solution to condition the prediction on the used solvent. Empirical results show performance increase relative to SOTA.

---

## Strengths and Weaknesses

### Strengths

- Predicting NMR spectra is a relevant problem in multiple applications of analytical chemistry, including organic chemistry research, the pharmaceutical industry (assessment of impurities at production), identifying environmental pollutants, and so on.
- The suggested method utilizes a vast amount of untapped literature data.
- It addresses the problem of solvent effects.

### Weaknesses

- No error bars are provided in any of the result tables; it is very hard to tell if the differences are significant or not.
- Many elements of Table 2 are left unfilled, but the reviewer fails to see the reason why they cannot be computed. Any prediction method can be applied to the structures in the ShiftDB-Lit database, and the \(L_{\mathrm{mol}}\) loss can be computed on the prediction. (The reviewer may be missing something—in that case the authors should clarify—but they see no reason why the missing values cannot be computed in Table 2.)
- The idea of using a set-based supervision loss is not particularly novel.

**Note from reviewer:** Addressing the evaluation problem (adding error bars, providing all losses in Table 2 that can be computed) would automatically result in +1 in their overall score.

---

## Scores

| Criterion     | Score | Label     |
|---------------|-------|-----------|
| Soundness     | 3     | good      |
| Presentation  | 4     | excellent |
| Significance  | 4     | excellent |
| Originality   | 2     | fair      |

---

## Key Questions for Authors

1. Please provide error bars for MAE and RMSE values, and provide the missing values in Table 2.
2. The term “unsupervised” is used many times in the paper, but the literature data is not unsupervised as there is a label—a set label. As properly named elsewhere, this is a weakly supervised setting. To avoid confusion, change occurrences of “unsupervised” to “weakly supervised” in the text; “semi-supervised” can also be changed to “weakly supervised” (except for the title, which cannot be changed anymore).
3. It seems the authors ignore multiplicity of \(^1\)H NMR peaks altogether. This is easy to use and valuable information to restrict possible permutations. Did they try using it?
4. On Figure 3, the model-collapse argument about the red line in the case of \(^1\)H seems fragile given that the \(^{13}\)C line does not show this behaviour. Nothing is mentioned about this in the text. Please elaborate on possible reasons.
5. How many conformers do you generate and use as input?

**Minor:** Using \(\hat{s}\) as ground truth and \(s\) as the prediction goes against traditional use, where a hat denotes the estimate; it would be more appropriate to switch the two notations.

---

## Limitations

No explicit discussion of limitations is provided. One that comes to mind is the difficulty of guessing the dominant conformer(s) in the sample, as that can depend on the solvent for example; furthermore there can be an ensemble of conformers contributing to the signal.

---

## Final Decisions

| Item | Value |
|------|--------|
| **Overall Recommendation** | **3: Weak reject** — A paper with clear merits, but also some weaknesses, which overall outweigh the merits. Papers in this category require revisions before they can be meaningfully built upon by others. Please use sparingly. |
| **Confidence** | **4** — You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. |
| **Compliance with LLM Reviewing Policy** | Affirmed |
| **Code of Conduct Acknowledgement** | Affirmed |
