# Linear Discriminant Analysis (LDA)

Advanced | classical-ml | classification | generative-models

### The problem, from first principles

The preceding logistic model learns a boundary directly. LDA instead describes where each class's points tend to lie and how they spread, then derives the boundary from those descriptions. Assuming both classes share one covariance is what makes that boundary linear.

### From theory to code

Implement `lda_fit(X, y)` by splitting classes 0 and 1, estimating their means and shared covariance, then deriving the returned score vector and intercept.

### Constraints

- `X` contains feature rows and `y` contains binary labels `0` and `1`.
- Return `w` with one entry per feature and scalar `b`.
- Pool the two centered scatter matrices and divide by `n - 2`.
- Invert the pooled covariance with `np.linalg.inv`.
- Include the class-prior log ratio in `b`; unequal class sizes must affect the boundary.

### Hints

Open one at a time. Each gives away a little more than the last.

<details><summary>Hint 1</summary>

Boolean masks `y == 0` and `y == 1` separate the two populations without a loop.

</details>

<details><summary>Hint 2</summary>

Compute each class's centered scatter with `(Xk - muk).T @ (Xk - muk)`, then pool before inverting.

</details>
