import numpy as np

def solve(X):
    """Implement missing-value-mean-imputation from the mathematical contract in README.md."""
    X = np.array(X)
    n_col = X.shape[-1]
    for i in range(n_col):
        my_arr = X[:,i]
        if np.isnan(my_arr).sum() == 0:
            continue
        i_mean = np.nanmean(my_arr)
        my_arr = np.where(np.isnan(my_arr), i_mean, my_arr)
        X[:,i] = my_arr
    return X
