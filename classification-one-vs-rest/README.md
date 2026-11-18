# Multiclass via One-vs-Rest, contrasted against Softmax

Intermediate | classical-ml | classification

### The problem, from first principles

`Stretch: Softmax + Categorical Cross-Entropy` handles multiclass classification with one unified model: a single weight matrix, one softmax, trained jointly on all classes at once. But `Full Training Loop`'s binary logistic regression came first, and it turns out you don't strictly NEED softmax to handle more than two classes at all, a much simpler idea works too: train a SEPARATE binary "is it this class, or not" classifier for every class, independently, and at prediction time just ask all of them and go with whichever is most confident.

This is One-vs-Rest (also called One-vs-All), and building it here, entirely out of `Full Training Loop`'s own binary machinery, makes the contrast with Softmax's single joint model concrete rather than abstract: two genuinely different ways of solving the same multiclass problem, with real, different tradeoffs.

### From theory to code

Theory trains `num_classes` independent binary classifiers by reusing `Full Training Loop`'s `train_logistic_regression` once per class (each time treating one class as positive, everything else as negative), then predicts by comparing all classifiers' confidence scores and picking the winner.

Implement `train_one_vs_rest(input, target, num_classes, lr, epochs)` first, then `predict_one_vs_rest(input, weights, biases)` on top of it.

### Constraints

- Each binary classifier is trained completely independently (no shared parameters, no joint loss).
- `weights` is `(num_classes, in_features)`, `biases` is `(num_classes,)`, stacked in the shape `linear`'s general multi-output convention already expects.
- Prediction picks the class with the single highest score, no thresholding at `0.5`.

### Hints

Open one at a time. Each gives away a little more than the last.

<details>
<summary>Hint 1</summary>

For each class `k`, build a binary target array: `1` where `target == k`, `0` everywhere else, then call `train_logistic_regression` on that binary target exactly like `Full Training Loop` already does.

</details>

<details>
<summary>Hint 2</summary>

`linear(input, weights, biases)` with `weights` shaped `(num_classes, in_features)` computes every class's raw score for every input row in one matmul, exactly the general multi-output convention `Hypothesis Function` was built for.

</details>
