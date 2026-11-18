import numpy as np


def soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    The soft-thresholding operator, the core building block of Lasso's
    solver: shrinks x toward zero by `threshold`, snapping anything
    within `threshold` of zero to EXACTLY zero.

        soft_threshold(x, t) = sign(x) * max(|x| - t, 0)
    """
    softened = np.sign(x) * np.maximum( np.abs(x) - threshold, 0)
    return softened


def lasso_regression_coordinate_descent(
    input: np.ndarray, target: np.ndarray, alpha: float = 1.0, epochs: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ridge Regression (L2) has a clean closed form; Lasso's L1 penalty
    does NOT (it's not differentiable at exactly 0), so this uses
    coordinate descent instead: repeatedly update ONE weight at a
    time, holding every other weight fixed, using soft_threshold to
    solve that single-coordinate subproblem exactly, cycling through
    all coordinates for several epochs.

    Center `input` and `target` first (subtract their means) so the
    bias can be recovered afterward without needing to include it in
    the penalized coordinate descent loop at all.

    Returns (weight, bias) in the same (1, in_features)/(1,) shapes
    the other regression questions in this curriculum use.
    """
    n, d = input.shape
    
    X_mean = input.mean( axis = 0 )
    y_mean = target.mean()
    X_centered = input - X_mean 
    y_centered = target - y_mean
    weight = np.zeros(d)
    
    for _ in range(epochs):
        for j in range(d):
            residual = y_centered - X_centered @ weight + X_centered[:, j] * weight[j]
            rho_j = (X_centered[:, j] @ residual ) / n
            z_j = (X_centered[:, j] @ X_centered[:, j]) / n
            weight[j] = soft_threshold(rho_j, alpha) / z_j if z_j > 0 else 0.0
    
    bias = y_mean - X_mean @ weight
    
    return weight.reshape(1,-1), np.array([bias])
