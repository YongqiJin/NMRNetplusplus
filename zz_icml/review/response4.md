# Response to Reviewer pNG7 (review4)

## Weaknesses

> The ShiftDB-Lit dataset cannot be considered as a contribution of this paper, as it is simply a filtered version of the original dataset[1] using a systematic three-stage process.
> [1] Nmrextractor: leveraging large language models to construct an experimental nmr database from open-source scientific publications

[TODO: response]

> In Line 209, the authors formulate the weakly-supervised (molecule-level) loss as a bipartite matching loss and obtain the minimum by sorting the predicted and observed shifts and matching them in order. I am curious whether the sorting is done in ascending or descending order, and whether this choice would affect the final result.

[TODO: response]

> In the experiments section, the authors should specify which datasets are used for supervised and weakly‑supervised training, along with their respective data ratios.

[TODO: response]

> Table 2 is confusing. What does the superscript “3” indicate? Why are there so many blank entries? Moreover, the baseline setting is unfair. As the authors point out, the ~60% improvement is largely due to the baseline being evaluated under an OOD test.

[TODO: response]

> I strongly suggest that the authors include more convincing baselines to demonstrate the effectiveness of the semi‑supervised training framework. For instance, first training with the weakly‑supervised molecule‑level loss and then fine‑tuning with the supervised atom‑level loss would help validate the carefully designed loss function. Additionally, an unsupervised baseline using a common masking strategy would also be a strong baseline.

[TODO: response]

> I am confused about why 3D molecular conformations are generated in Line 130, as I could not find any experiments that actually make use of this 3D information.

[TODO: response]

## Limitations

> This paper does not discuss its limitations.

[TODO: response]
