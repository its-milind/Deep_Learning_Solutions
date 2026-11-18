import numpy as np


def ridge_regression_closed_form(
    input: np.ndarray, target: np.ndarray, alpha: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    
    n = input.shape[0]
    
    X = np.hstack([input, np.ones((n,1))])
    y = target.reshape(-1, 1)
    
    I = np.eye(X.shape[1])
    I[-1,-1] = 0
    theta = np.linalg.solve(X.T @ X + alpha * I, X.T @ y)
    weights = theta[:-1].T
    bias = theta[-1]
    return weights, bias
