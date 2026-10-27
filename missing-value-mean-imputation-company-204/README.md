# missing-value-mean-imputation — Microsoft case

Beginner | problemset | data-stats-for-ds | data-cleaning | microsoft

This is a hypothetical engineering scenario inspired by the kind of work a **Microsoft** ranking and experimentation team might handle; it is not a real interview question or a claim that Microsoft uses this exact task. The team needs a reliable implementation for feature cleaning in a production-oriented ML workflow. A small implementation error can change the model input or metric enough to make a downstream result misleading.

Replace NaN entries in each feature column with that column's mean computed from the observed values.

Process each column independently: compute the mean of its observed (non-NaN) values and put it in place of every `NaN` in that column. A column with no observed value stays `NaN`.

Implement `solve(X)`.

**Returns.** Return a new float NumPy matrix of the same shape. The input is not modified.

### Examples

**Example 1**

Input:

```python
solve([[1.0, float('nan')], [3.0, 5.0]])
```

Output:

```text
[[1.0, 5.0], [3.0, 5.0]]
```

**Example 2**

Input:

```python
solve([[1.0, 2.0], [float('nan'), 4.0], [5.0, float('nan')]])
```

Output:

```text
[[1.0, 2.0], [3.0, 4.0], [5.0, 3.0]]
```
