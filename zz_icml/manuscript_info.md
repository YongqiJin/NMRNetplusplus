# Submission metadata

## Title

From Human Labels to Literature: Semi-Supervised Learning of NMR Chemical Shifts at Scale

## Abstract

Accurate prediction of nuclear magnetic resonance (NMR) chemical shifts is fundamental to spectral analysis and molecular structure elucidation, yet existing machine learning methods rely on limited, labor-intensive atom-assigned datasets. We propose a semi-supervised framework that learns NMR chemical shifts from millions of literature-extracted spectra without explicit atom-level assignments, integrating a small amount of labeled data with large-scale unassigned spectra. We formulate chemical shift prediction from literature spectra as a permutation-invariant set supervision problem, and show that under commonly satisfied conditions on the loss function, optimal bipartite matching reduces to a sorting-based loss, enabling stable large-scale semi-supervised training beyond traditional curated datasets. Our models achieve substantially improved accuracy and robustness over state-of-the-art methods and exhibit stronger generalization on significantly larger and more diverse molecular datasets. Moreover, by incorporating solvent information at scale, our approach captures systematic solvent effects across common NMR solvents for the first time. Overall, our results demonstrate that large-scale unlabeled spectra mined from the literature can serve as a practical and effective data source for training NMR shift models, suggesting a broader role of literature-derived, weakly structured data in data-centric AI for science.

## Primary area

Applications → Chemistry, Physics, and Earth Sciences

## Keywords

- Computational chemistry
- Data-centric machine learning
- Semi-supervised learning
