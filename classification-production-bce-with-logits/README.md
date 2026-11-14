# Production Engineering: Fused, Numerically-Stable Loss

Advanced | classical-ml | classification | production-engineering | numerical-stability

### The problem, from first principles

Computing sigmoid first can round extreme logits to exact probabilities of zero or one before BCE sees them. A production loss avoids that irreversible step: it scores raw logits directly using an equivalent expression whose exponential never grows.

### From theory to code

Implement `bce_with_logits_loss(z, y)` without calling sigmoid. Theory gives the fused per-sample expression and explains why the exponent must receive a non-positive value.

### Constraints

- `z` and binary `y` have matching shapes.
- Return the mean loss as a Python `float`.
- Never create a sigmoid probability array or call `exp` on a positive value.
- Use `np.maximum(z, 0)`, `np.log1p`, and `np.exp(-np.abs(z))`.
- Remain finite for extreme positive and negative logits.

### Hints

Open one at a time. Each gives away a little more than the last.

<details><summary>Hint 1</summary>

Separate the expression into a piecewise-linear correction and a small logarithmic remainder.

</details>

<details><summary>Hint 2</summary>

`-np.abs(z)` is never positive; take its exponential, then use `log1p` for the logarithmic term.

</details>
