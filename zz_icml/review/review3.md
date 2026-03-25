# Official Review of Submission2786 by Reviewer wmHV

**Official Review** by Reviewer wmHV  
**Date:** 13 Mar 2026, 09:22 (modified: 24 Mar 2026, 20:25)  
**Visibility:** Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer wmHV

---

## Summary

This paper proposes a semi-supervised framework for NMR chemical shift prediction that combines a small atom-assigned dataset (NMRShiftDB2, ~13–27k molecules) with ~900k unassigned literature-extracted spectra (ShiftDB-Lit). Learning from unassigned spectra is formulated as permutation-invariant set supervision; the authors prove that for convex monotonically increasing losses, optimal bipartite matching reduces to sorting-based pairing, avoiding the \(O(n^3)\) Hungarian assignment.

The model builds on the SE(3)-equivariant NMRNet architecture with solvent conditioning via CLS-token embedding, treating solvent as a global contextual bias on shifts. It reports 13–20% MAE reductions on NMRShiftDB2 for \(^1\)H and \(^{13}\)C, larger gains on ShiftDB-Lit, and strong gains for underrepresented solvents (e.g. DMSO-d\(_6\)). Initial heteroatom baselines (\(^{19}\)F, \(^{31}\)P, \(^{11}\)B, \(^{29}\)Si) are also given.

An ablation shows model collapse when training only on unassigned data: shift *sets* match the spectrum collectively but are assigned to the wrong atoms—molecule-level loss falls while atom-level accuracy degrades. A moderate amount of atom-assigned data is needed to anchor per-atom correctness under the sorting-based loss.

---

## Strengths and Weaknesses

### Strengths (S1–S6)

1. **S1 (Soundness).** The sorting-based loss equivalence is cleanly motivated and formally proven. The proof in Appendix F proceeds through a convexity/monotonicity lemma (Lemma F.1) and a swapping argument (Theorem F.2) that transforms any permutation into the sorted matching without increasing loss. The conditions—that the loss has the form \(\ell(x,y)=f(|x-y|)\) with \(f\) monotonically increasing and convex—are mild and satisfied by MAE, MSE, and Huber loss, so the result is broadly applicable rather than architecture-specific.

2. **S2 (Soundness).** The ablation study (Table 5) is well-designed. The five training configurations isolate contributions: Exp. 1 (supervised on NMRShiftDB2 only) yields \(^1\)H MAE 0.1972 and \(^{13}\)C MAE 1.1518; Exp. 2 (weakly supervised on ShiftDB-Lit only) yields \(^1\)H MAE 0.2412 and \(^{13}\)C MAE 1.5214, showing model collapse; Exp. 4 (supervised + weakly supervised, both on NMRShiftDB2) yields \(^1\)H MAE 0.2152 and \(^{13}\)C MAE 1.1503, indicating no benefit from weak supervision on the same labeled data; Exp. 5 (supervised on NMRShiftDB2 + weakly supervised on ShiftDB-Lit) achieves the best results (\(^1\)H MAE 0.1709, \(^{13}\)C MAE 0.9270). Reporting the collapse case (Exp. 2), not only the best case, strengthens trust in the methodology.

3. **S3 (Soundness).** Cross-solvent validation (Table 9) is careful. On the CDCl\(_3\)/DMSO-d\(_6\) pair (419 molecules), correct solvent yields \(^1\)H MAE 0.0832 vs. 0.2814 (wrong) and 0.1228 (none). The same pattern across six solvent-pair settings—correct tokens \(\Rightarrow\) lowest error—argues against solvent conditioning being only a global bias.

4. **S4 (Originality).** Framing unassigned NMR spectra as permutation-invariant set supervision is natural but underused. Sorting-style losses appear elsewhere in ML; proving optimality under stated conditions for this setting (rather than treating sorting as a heuristic) is a substantive contribution.

5. **S5 (Significance).** Gains on NMRShiftDB2 (\(^1\)H MAE 0.1972\(\to\)0.1709; \(^{13}\)C MAE 1.1518\(\to\)0.9270) come **without** architectural changes to NMRNet, isolating data and loss design. **ShiftDB-Lit** (898,422 \(^1\)H and 704,373 \(^{13}\)C entries) is roughly **26–70×** larger than NMRShiftDB2 and materially expands training resources for the field.

6. **S6 (Presentation).** Structure is clear. Figure 1 distinguishes atom-level vs. molecule-level supervision. The move from general bipartite matching (Eq. 2) to the sorting simplification (Eq. 4) is paced well.

### Weaknesses (W1–W6)

1. **W1 (Major Soundness).** The ShiftDB-Lit test set evaluation conflates two effects. The NMRNet baseline is trained only on NMRShiftDB2, making ShiftDB-Lit out-of-distribution for it, while the semi-supervised model trains on ShiftDB-Lit, making it in-distribution. The reported reductions ¹H MAE from 0.1395 to 0.0559 (↓59.9%) and ¹³C MAE from 1.2591 to 0.5060 (↓59.8%) therefore reflect both the value of semi-supervised learning and simple distribution coverage. The authors acknowledge this (lines 283–293) but do not attempt to disentangle the two contributions for instance, by evaluating on a held-out scaffold split that is OOD for both models. This is fixable: a scaffold-based partition of ShiftDB-Lit that excludes training scaffolds from both models would isolate the semi-supervised learning gain.

2. **W2 (Major Soundness).** ShiftDB-Lit uses a random 4:1 train/test split (Section 4.1) rather than a scaffold split. For a dataset of ~1.6M molecules, random splitting almost certainly places structurally similar molecules on both sides of the partition, inflating reported metrics. The NMRShiftDB2 results use a pre-defined benchmark split and are less affected, but the ShiftDB-Lit numbers which show the most dramatic gains (59.9% and 59.8% MAE reductions) and carry the paper's strongest narrative claims are the most vulnerable to this concern. This is also fixable by re-evaluating with a scaffold split.

3. **W3 (Major Soundness).** The solvent embedding is limited to three categories: CDCl₃ (89.1% of data, 162,509 test molecules), DMSO-d₆ (5.7%, 11,623 test molecules), and "others" (5,553 test molecules for ¹H). Table 3 shows the catch-all "others" category retains the highest MAE (0.0996 for ¹H, 0.8684 for ¹³C with solvent incorporation), and its improvement over the solvent-agnostic baseline is modest (↓9.5% for ¹H, ↓9.0% for ¹³C) compared to the gains for DMSO-d₆ (↓46.8% for ¹H, ↓17.8% for ¹³C). There is no analysis of (a) whether the "others" embedding does meaningful work beyond what the model learns without it, or (b) whether finer solvent granularity (e.g., 5–10 categories) would improve performance. Given that ShiftDB-Lit presumably contains solvent identity metadata beyond these three bins, the coarse grouping appears to be an underexplored design choice rather than a hard constraint. A targeted ablation comparing 3-bin vs. finer-grained solvent embeddings would address this.

4. **W4 (Minor Soundness).** The \(\lambda\) sweep (Figure 3) uses powers of two from \(2^0\) to \(2^7\); the optimum is \(\lambda=16\) (\(2^4\)). On NMRShiftDB2 (\(L_{\mathrm{atom}}\)), \(^1\)H MAE is ~0.170 at \(\lambda=16\) vs. ~0.190 at \(\lambda=128\); \(^{13}\)C MAE ~0.93 vs. ~1.15 over the same range. A finer grid near the optimum or a sensitivity curve would show how sharp the basin is. Easy to add.

5. **W5 (Minor Presentation).** The heteroatom results (Table 4) report MAE values of 2.2809 ppm (¹⁹F), 1.3099 ppm (³¹P), 0.8287 ppm (¹¹B), and 1.9186 ppm (²⁹Si), with R² values of 0.7216, 0.9634, 0.9406, and 0.8901, respectively. However, these are presented without any baselines or comparison methods, making it difficult to contextualize whether these numbers represent strong or weak performance. Even a simple DFT or HOSE comparison would anchor them. The ¹⁹F R² of 0.7216 in particular seems relatively low and warrants discussion.

6. **W6 (Minor Presentation).** Wall-clock training time is not reported. Table 7 shows 10 epochs for semi-supervised vs. 50 for supervised, with batch sizes of 4 (labeled) + 16 (unlabeled) for semi-supervised vs. 8 (labeled only) for supervised, and learning rates of 4e-4 vs. 1e-4 respectively. Since the weakly-supervised data contains 898,422 ¹H and 704,373 ¹³C entries compared to NMRShiftDB2's 12,800 ¹H and 26,859 ¹³C entries (~26-70× larger), the actual computational cost is difficult to assess from epoch counts alone. This matters for practitioners deciding whether to adopt the method.

---

## Scores

- **Soundness:** 3: good
- **Presentation:** 3: good
- **Significance:** 3: good
- **Originality:** 3: good

---

## Key Questions For Authors

1. Can you evaluate on a scaffold-split held-out set from ShiftDB-Lit that is OOD for both the baseline and semi-supervised model? This would directly address W1 and W2 by isolating the semi-supervised learning gain from distribution coverage.
2. What is the solvent distribution within the “others” category (5,553 \(^1\)H test molecules, 5,485 \(^{13}\)C test molecules), and have you tried finer granularity? The ↓9.5% (\(^1\)H) and ↓9.0% (\(^{13}\)C) improvements for “others” vs. ↓46.8% and ↓17.8% for DMSO-d\(_6\) suggest the single embedding may be underperforming. (Addresses W3.)
3. What are the wall-clock training times for the supervised (50 epochs, batch size 8) vs. semi-supervised (10 epochs, batch sizes 4+16) settings on the reported NVIDIA RTX 4090? (Addresses W6.)
4. For \(^{19}\)F, the \(R^2\) of 0.7216 is notably lower than for \(^{31}\)P (0.9634) and \(^{11}\)B (0.9406). Can you comment on whether this reflects intrinsic difficulty, data noise, or the broader chemical shift range (~300 to 300 ppm per Table 6)?

---

## Limitations

Yes.

---

## Final Decisions

- **Overall Recommendation:** 5: Accept: Technically solid paper, with high impact on at least one sub-area of AI or moderate-to-high impact on more than one area of AI, with good-to-excellent evaluation, resources, reproducibility, and no unaddressed ethical considerations.
- **Confidence:** 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
- **Compliance With LLM Reviewing Policy:** Affirmed
- **Code Of Conduct Acknowledgement:** Affirmed
