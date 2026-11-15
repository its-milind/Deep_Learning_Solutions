# LogSoftmax + NLLLoss: the two pieces CrossEntropyLoss actually fuses

Intermediate | classical-ml | classification | loss-functions

### The problem, from first principles

`Stretch: Softmax + Categorical Cross-Entropy` computed `softmax`, then took `log` of the true class's probability, two separate operations chained together. That two-step version has a real, silent failure mode: when a logit dominates strongly enough, `softmax` correctly produces a probability extremely close to (but not exactly) `0` for the losing classes, and `log` of a number that close to zero either underflows to `-inf` or, worse, rounds to EXACTLY `0.0` in floating point, at which point `log(0.0)` is `-inf`, propagating `nan` through the rest of the computation.

`torch.nn.functional.cross_entropy` never actually computes `log(softmax(Z))` as two separate steps for exactly this reason. It's built from two pieces, `log_softmax` (which never actually forms the raw probabilities, staying in log-space the entire time) and `nll_loss` (which just indexes and averages), and this question builds both pieces to see precisely how the fused version sidesteps the naive version's numerical trap.

### From theory to code

Theory derives `log_softmax` algebraically (expanding `log(exp(z_i) / sum(exp(z_j)))` and simplifying) into a form that never divides two floating-point numbers close to zero, and defines `nll_loss` as a simple indexing-and-averaging operation on whatever log-space values it's handed.

Implement `log_softmax(Z)` first, then `nll_loss(log_probs, y_indices)` on top of it.

### Constraints

- `log_softmax` must not call `np.log(softmax(Z))` as two separate steps, it must derive the log-space result directly, algebraically.
- `Z` is `(n, num_classes)`, `y_indices` is `(n,)`, integer class indices.
- `nll_loss` averages over rows (the `n` axis), consistent with `06-softmax-cce`'s own `cce_loss` reduction convention.

### Hints

Open one at a time. Each gives away a little more than the last.

<details>
<summary>Hint 1</summary>

Start from `06-softmax-cce`'s own max-shift trick (`Z - max(Z, axis=1, keepdims=True)`), then subtract `log(sum(exp(shifted Z)))` instead of dividing and taking a separate log.

</details>

<details>
<summary>Hint 2</summary>

`nll_loss` is one line: `-np.mean(log_probs[np.arange(n), y_indices])`, indexing out each row's true-class log-probability.

</details>
