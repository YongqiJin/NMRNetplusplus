# Response to Reviewer iTNC (review1)

## Weaknesses / concerns

> My main concern is evaluation fairness and interpretation. The paper emphasizes large gains on ShiftDB-Lit, but the baseline is trained only on NMRShiftDB2 while the proposed model leverages ShiftDB-Lit during training, so the comparison is partly OOD-vs-ID rather than purely method-vs-method. The paper does acknowledge this, but the framing should be more careful.

[TODO: response]

> A second concern is novelty level. The application is valuable, and the sorting reduction is elegant, but the overall method is still a relatively direct semi-supervised extension of an existing backbone rather than a fundamentally new model family. The strongest novelty is really the formulation plus the scale of the data resource.

[TODO: response]

> A third concern is data quality / noise robustness. The literature-extracted dataset is large, but inevitably noisy. The paper describes filtering procedures, which is good, but the main text does not quantify error rates from extraction, OCSR, parsing, solvent normalization, or duplicate handling. Since the paper’s central claim relies on learning from noisy literature data, this deserves more explicit auditing.

[TODO: response]

> The solvent modeling is promising but still fairly coarse. Solvents are grouped into three categories, with “others” collapsed into one embedding. That is understandable for data imbalance reasons, but it limits chemical interpretability and may hide meaningful solvent-specific behavior.

[TODO: response]

> Finally, the paper would benefit from stronger baselines in the weak-supervision setting. Most comparisons are against older supervised predictors or the NMRNet baseline. There is less discussion of alternative set-prediction / permutation-invariant training objectives or semi-supervised baselines beyond the chosen formulation.

[TODO: response]

## Key Questions for Authors

> Can the authors quantify the noise level of ShiftDB-Lit more directly, for example by manually auditing a random subset of extracted examples?

[TODO: response]

> How much of the gain comes from the sorting-based weak loss itself versus simply exposing the model to much broader chemical coverage?

[TODO: response]

> Did the authors deduplicate near-identical molecules or control for overlap between literature-derived molecules and benchmark molecules?

[TODO: response]

> Could the authors compare against a Hungarian-loss implementation directly on a smaller subset to verify that the sorting surrogate is empirically equivalent in practice, not only theoretically?

[TODO: response]

> For solvent modeling, what happens if the most common additional solvents are modeled separately rather than merged into “others”?

[TODO: response]

## Limitations

> The paper would be stronger with one extra section on data auditing: extraction quality, solvent label normalization, duplicates, and error examples. It would also help to separate the claims more clearly:
> - “semi-supervised objective is effective,”
> - “large-scale literature data improves coverage,”
> - “solvent conditioning adds value.”

> Right now these are somewhat intertwined. I would also like a clearer comparison to other possible permutation-invariant objectives and perhaps a discussion of when the sorting reduction fails if the loss assumptions are violated.

[TODO: response]
