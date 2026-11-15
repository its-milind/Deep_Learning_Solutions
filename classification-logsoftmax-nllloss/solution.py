import numpy as np


def log_softmax(Z: np.ndarray) -> np.ndarray:
    max_subbed = Z - np.max(Z, axis = 1, keepdims = True)
    log_softmax = max_subbed - np.log( np.sum( np.exp( max_subbed ), axis =1 , keepdims = True ) ) 
    return log_softmax

def nll_loss(log_probs: np.ndarray, y_indices: np.ndarray) -> float:
    n = log_probs.shape[0]
    nll_loss = -np.mean( log_probs[np.arange(n), y_indices])
    return float(nll_loss)
