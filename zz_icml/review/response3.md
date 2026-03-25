# Response to Reviewer wmHV (review3)

## Weaknesses (W1–W6)

> **W1 (Major Soundness).** ShiftDB-Lit test evaluation conflates two effects: the NMRNet baseline is trained only on NMRShiftDB2 (ShiftDB-Lit is OOD), while the semi-supervised model trains on ShiftDB-Lit (ID). Reported drops—\(^1\)H MAE 0.1395\(\to\)0.0559 (↓59.9%), \(^{13}\)C MAE 1.2591\(\to\)0.5060 (↓59.8%)—mix semi-supervised value with distribution coverage. The paper acknowledges this (lines 283–293) but does not disentangle them (e.g. via a scaffold held-out set that is OOD for **both** models). A scaffold partition excluding training scaffolds for both models would isolate the semi-supervised gain.

[TODO: response]

> **W2 (Major Soundness).** ShiftDB-Lit uses a **random 4:1** train/test split (Section 4.1), not a scaffold split. At ~1.6M molecules, random splits likely leak similar structures across train/test and **inflate** metrics. NMRShiftDB2 uses a fixed benchmark split; ShiftDB-Lit numbers (where the story is strongest) are most exposed. Re-evaluation under a scaffold split would address this.

[TODO: response]

> **W3 (Major Soundness).** Solvent is **three** buckets: CDCl\(_3\) (89.1%; 162,509 test mols), DMSO-d\(_6\) (5.7%; 11,623), and “others” (5,553 test mols for \(^1\)H). Table 3: “others” has the highest MAE (0.0996 \(^1\)H, 0.8684 \(^{13}\)C with solvent); gains vs. solvent-agnostic are modest (↓9.5% \(^1\)H, ↓9.0% \(^{13}\)C) vs. DMSO-d\(_6\) (↓46.8% \(^1\)H, ↓17.8% \(^{13}\)C). There is little analysis of (a) whether “others” helps beyond the no-solvent model or (b) whether finer bins (e.g. 5–10 solvents) help. Metadata likely supports finer grouping; a 3-bin vs. finer-embedding ablation would clarify.

[TODO: response]

> **W4 (Minor Soundness).** The \(\lambda\) sweep (Figure 3) uses powers of two from \(2^0\) to \(2^7\); the optimum is \(\lambda=16\) (\(2^4\)). On NMRShiftDB2 (\(L_{\mathrm{atom}}\)), \(^1\)H MAE is ~0.170 at \(\lambda=16\) vs. ~0.190 at \(\lambda=128\); \(^{13}\)C MAE ~0.93 vs. ~1.15 over the same range. A finer grid near the optimum or a sensitivity curve would show how sharp the basin is. Easy to add.

[TODO: response]

> **W5 (Minor Presentation).** Heteroatoms (Table 4): MAE 2.2809 ppm (\(^{19}\)F), 1.3099 ppm (\(^{31}\)P), 0.8287 ppm (\(^{11}\)B), 1.9186 ppm (\(^{29}\)Si); \(R^2\) 0.7216, 0.9634, 0.9406, 0.8901. **No baselines**—hard to judge absolute quality. Even DFT or HOSE-style references would help. The low \(^{19}\)F \(R^2\) (0.7216) especially needs discussion.

[TODO: response]

> **W6 (Minor Presentation).** **Wall-clock time** is omitted. Table 7: 10 epochs (semi-supervised) vs. 50 (supervised); batch 4+16 vs. 8; LR \(4\times10^{-4}\) vs. \(1\times10^{-4}\). Weakly supervised pools (898,422 \(^1\)H / 704,373 \(^{13}\)C) vs. NMRShiftDB2 (12,800 \(^1\)H / 26,859 \(^{13}\)C), ~**26–70×** larger—epoch counts alone do not reflect cost. Practitioners need timing to compare adoption cost.

[TODO: response]

## Key Questions For Authors

> Can you evaluate on a scaffold-split held-out set from ShiftDB-Lit that is OOD for both the baseline and semi-supervised model? This would directly address W1 and W2 by isolating the semi-supervised learning gain from distribution coverage.

[TODO: response]

> What is the solvent distribution within the “others” category (5,553 \(^1\)H test molecules, 5,485 \(^{13}\)C test molecules), and have you tried finer granularity? The ↓9.5% (\(^1\)H) and ↓9.0% (\(^{13}\)C) improvements for “others” vs. ↓46.8% and ↓17.8% for DMSO-d\(_6\) suggest the single embedding may be underperforming. (Addresses W3.)

[TODO: response]

> What are the wall-clock training times for the supervised (50 epochs, batch size 8) vs. semi-supervised (10 epochs, batch sizes 4+16) settings on the reported NVIDIA RTX 4090? (Addresses W6.)

[TODO: response]

> For \(^{19}\)F, the \(R^2\) of 0.7216 is notably lower than for \(^{31}\)P (0.9634) and \(^{11}\)B (0.9406). Can you comment on whether this reflects intrinsic difficulty, data noise, or the broader chemical shift range (~300 to 300 ppm per Table 6)?

[TODO: response]
