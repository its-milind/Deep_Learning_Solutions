import numpy as np


def softmax(Z: np.ndarray) -> np.ndarray:
    Z = Z-Z.max(axis = 1, keepdims = True)
    Z = np.exp(Z)
    Z = Z/Z.sum(axis = 1, keepdims = True)
    return Z


def cce_loss(P: np.ndarray, y_indices: np.ndarray) -> float:
    n = P.shape[0]
    classes = P[np.arange(n),y_indices]
    classes = np.clip(classes, 1e-12, 1.0)
    classes = -np.mean(np.log(classes))
    return float(classes)
