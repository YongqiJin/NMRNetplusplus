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

1. **W1 (Major Soundness).** ShiftDB-Lit test evaluation conflates two effects: the NMRNet baseline is trained only on NMRShiftDB2 (ShiftDB-Lit is OOD), while the semi-supervised model trains on ShiftDB-Lit (ID). Reported drops—\(^1\)H MAE 0.1395\(\to\)0.0559 (↓59.9%), \(^{13}\)C MAE 1.2591\(\to\)0.5060 (↓59.8%)—mix semi-supervised value with distribution coverage. The paper acknowledges this (lines 283–293) but does not disentangle them (e.g. via a scaffold held-out set that is OOD for **both** models). A scaffold partition excluding training scaffolds for both models would isolate the semi-supervised gain.

2. **W2 (Major Soundness).** ShiftDB-Lit uses a **random 4:1** train/test split (Section 4.1), not a scaffold split. At ~1.6M molecules, random splits likely leak similar structures across train/test and **inflate** metrics. NMRShiftDB2 uses a fixed benchmark split; ShiftDB-Lit numbers (where the story is strongest) are most exposed. Re-evaluation under a scaffold split would address this.

3. **W3 (Major Soundness).** Solvent is **three** buckets: CDCl\(_3\) (89.1%; 162,509 test mols), DMSO-d\(_6\) (5.7%; 11,623), and “others” (5,553 test mols for \(^1\)H). Table 3: “others” has the highest MAE (0.0996 \(^1\)H, 0.8684 \(^{13}\)C with solvent); gains vs. solvent-agnostic are modest (↓9.5% \(^1\)H, ↓9.0% \(^{13}\)C) vs. DMSO-d\(_6\) (↓46.8% \(^1\)H, ↓17.8% \(^{13}\)C). There is little analysis of (a) whether “others” helps beyond the no-solvent model or (b) whether finer bins (e.g. 5–10 solvents) help. Metadata likely supports finer grouping; a 3-bin vs. finer-embedding ablation would clarify.

4. **W4 (Minor Soundness).** The \(\lambda\) sweep (Figure 3) uses powers of two from \(2^0\) to \(2^7\); the optimum is \(\lambda=16\) (\(2^4\)). On NMRShiftDB2 (\(L_{\mathrm{atom}}\)), \(^1\)H MAE is ~0.170 at \(\lambda=16\) vs. ~0.190 at \(\lambda=128\); \(^{13}\)C MAE ~0.93 vs. ~1.15 over the same range. A finer grid near the optimum or a sensitivity curve would show how sharp the basin is. Easy to add.

5. **W5 (Minor Presentation).** Heteroatoms (Table 4): MAE 2.2809 ppm (\(^{19}\)F), 1.3099 ppm (\(^{31}\)P), 0.8287 ppm (\(^{11}\)B), 1.9186 ppm (\(^{29}\)Si); \(R^2\) 0.7216, 0.9634, 0.9406, 0.8901. **No baselines**—hard to judge absolute quality. Even DFT or HOSE-style references would help. The low \(^{19}\)F \(R^2\) (0.7216) especially needs discussion.

6. **W6 (Minor Presentation).** **Wall-clock time** is omitted. Table 7: 10 epochs (semi-supervised) vs. 50 (supervised); batch 4+16 vs. 8; LR \(4\times10^{-4}\) vs. \(1\times10^{-4}\). Weakly supervised pools (898,422 \(^1\)H / 704,373 \(^{13}\)C) vs. NMRShiftDB2 (12,800 \(^1\)H / 26,859 \(^{13}\)C), ~**26–70×** larger—epoch counts alone do not reflect cost. Practitioners need timing to compare adoption cost.

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
