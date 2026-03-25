# Response to Reviewer 6gyS (review2)

## Weaknesses

> No error bars are provided in any of the result tables; it is very hard to tell if the differences are significant or not.

[TODO: response]

> Many elements of Table 2 are left unfilled, but the reviewer fails to see the reason why they cannot be computed. Any prediction method can be applied to the structures in the ShiftDB-Lit database, and the \(L_{\mathrm{mol}}\) loss can be computed on the prediction. (The reviewer may be missing something—in that case the authors should clarify—but they see no reason why the missing values cannot be computed in Table 2.)

[TODO: response]

> The idea of using a set-based supervision loss is not particularly novel.

[TODO: response]

## Key Questions for Authors

> Please provide error bars for MAE and RMSE values, and provide the missing values in Table 2.

[TODO: response]

> The term “unsupervised” is used many times in the paper, but the literature data is not unsupervised as there is a label—a set label. As properly named elsewhere, this is a weakly supervised setting. To avoid confusion, change occurrences of “unsupervised” to “weakly supervised” in the text; “semi-supervised” can also be changed to “weakly supervised” (except for the title, which cannot be changed anymore).

[TODO: response]

> It seems the authors ignore multiplicity of \(^1\)H NMR peaks altogether. This is easy to use and valuable information to restrict possible permutations. Did they try using it?

[TODO: response]

> On Figure 3, the model-collapse argument about the red line in the case of \(^1\)H seems fragile given that the \(^{13}\)C line does not show this behaviour. Nothing is mentioned about this in the text. Please elaborate on possible reasons.

[TODO: response]

> How many conformers do you generate and use as input?

[TODO: response]

> **Minor:** Using \(\hat{s}\) as ground truth and \(s\) as the prediction goes against traditional use, where a hat denotes the estimate; it would be more appropriate to switch the two notations.

[TODO: response]

## Limitations

> No explicit discussion of limitations is provided. One that comes to mind is the difficulty of guessing the dominant conformer(s) in the sample, as that can depend on the solvent for example; furthermore there can be an ensemble of conformers contributing to the signal.

[TODO: response]
