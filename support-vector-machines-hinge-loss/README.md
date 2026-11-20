# Hinge loss

Beginner | classic-ml

### The problem, from first principles

`Binary Cross-Entropy Loss` measures HOW confident a prediction is, wrong: even a correct prediction that's only slightly confident still incurs some loss, and cross-entropy keeps pushing confidence toward the extremes forever (`-log(p)` never actually reaches `0`). Support Vector Machines take a fundamentally different view of what "good enough" means: once a prediction is not just correct but correct BY A COMFORTABLE MARGIN, stop caring, that example contributes exactly zero additional loss, no matter how much MORE confident it could theoretically become.

That's hinge loss. Its whole shape is built around one number, a margin of `1`, that separates "this example is fine, ignore it" from "this example needs more attention, either wrong or too close to the boundary."

### From theory to code

Theory uses `{-1, +1}` labels (not `{0, 1}`) specifically because the formula `1 - target * scores` needs the label's SIGN to flip the score correctly for negative examples, and defines the loss as whatever's left after subtracting the margin, clipped at zero.

Implement `hinge_loss(scores, target, reduction="mean")` against that reasoning. The signature and docstring are already in the editor.

### Constraints

- `target` is `{-1, +1}`, not `{0, 1}`.
- `scores` are raw, unbounded classifier outputs, not probabilities.
- Same `reduction` modes (`"mean"`, `"sum"`, `"none"`) the other losses in this curriculum use.

### Hints

Open one at a time. Each gives away a little more than the last.

<details>
<summary>Hint 1</summary>

`target * scores` is positive when the prediction's sign agrees with the label, and its MAGNITUDE measures how confidently correct it is.

</details>

<details>
<summary>Hint 2</summary>

`np.maximum(0.0, 1.0 - target * scores)` is the entire formula, no separate correctness check needed, the `max` with `0` handles it.

</details>
