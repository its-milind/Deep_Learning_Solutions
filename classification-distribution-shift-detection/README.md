# Production Engineering: detecting train/serve distribution shift

Advanced | classical-ml | classification | evaluation

### The problem, from first principles

`Generalization: train/val split and the generalization gap` measured whether a model generalizes from training data to a held-out set drawn from the SAME underlying distribution. A deployed model faces a harder, ongoing version of this problem: the real world doesn't hold still. User behavior shifts, a product changes, a new customer segment appears, and the LIVE data flowing into a deployed model can quietly drift away from the distribution it was trained on, degrading performance in a way that never shows up in any offline validation metric, because validation was computed before the drift ever happened.

The Population Stability Index (PSI) is the standard production-ML tool for catching this: a single number, computed continuously on live traffic, that flags when a feature's distribution has drifted meaningfully from what the model was trained to expect, before that drift necessarily shows up as a visible drop in business metrics.

### From theory to code

Theory bins both the training distribution and the live distribution using bin edges chosen from the TRAINING data's own quantiles (so training data is roughly evenly spread across bins by construction), then measures how differently the live data falls across those same bins, in a KL-divergence-like formula.

Implement `bin_proportions(values, bin_edges)` first, then `population_stability_index(train_values, live_values, num_bins=10)`, then `detect_distribution_shift(train_values, live_values, num_bins=10, threshold=0.2)`.

### Constraints

- Bin edges are derived from `train_values`' quantiles, with the outermost edges widened to `-inf`/`+inf`.
- Clip proportions away from exactly `0` before taking a log (the same reason `01-entropy`, Math & Statistics, clips).
- `detect_distribution_shift` uses the conventional threshold, `PSI > 0.2` signals a significant shift.

### Hints

Open one at a time. Each gives away a little more than the last.

<details>
<summary>Hint 1</summary>

`np.quantile(train_values, np.linspace(0, 1, num_bins + 1))` gives bin edges that split the TRAINING data into roughly equal-sized buckets.

</details>

<details>
<summary>Hint 2</summary>

`np.histogram(values, bins=bin_edges)` counts how many values fall in each bin; divide by `len(values)` to get proportions.

</details>
