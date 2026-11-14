# Stretch: Class Imbalance Handling

Intermediate | classical-ml | classification | class-imbalance | stretch

### The problem, from first principles

When one label is scarce, an unweighted average can let a classifier look good by prioritizing the frequent class. Per-class weights make mistakes on selected labels count more without changing which prediction is correct.

### From theory to code

Implement `weighted_bce_loss(p, y, class_weights)`: use the same clipped BCE terms as `02-bce-loss`, then multiply each sample by the weight selected from its actual class.

### Constraints

- `p` and binary `y` have matching sample shapes.
- `class_weights` provides keys `0` and `1`.
- Clip probabilities to `[1e-12, 1 - 1e-12]`.
- Select class-1 weight where `y == 1`, otherwise class-0 weight.
- Return the negative mean as a Python `float` without loops.

### Hints

Open one at a time. Each gives away a little more than the last.

<details><summary>Hint 1</summary>

First create an array of one weight per label; then it broadcasts naturally against one loss per sample.

</details>

<details><summary>Hint 2</summary>

`np.where(y == 1, class_weights[1], class_weights[0])` performs the per-sample lookup.

</details>
