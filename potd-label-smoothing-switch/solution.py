import numpy as np


def label_smoothing(logits: np.ndarray, y: int, eps: float) -> tuple[np.ndarray, float]:
    """
    Build the label-smoothed target distribution over C = len(logits)
    classes and compute its cross-entropy loss against logits.

    q[c] = 1 - eps + eps/C for c == y, else eps/C.
    loss = -sum(q * log_softmax(logits)).

    Return (q, loss).
    """
    # TODO: smooth EVERY class by eps/C first, then add the remaining
    # mass (1 - eps) onto the true class y.
    C = len(logits)
    q = np.full(C, eps/C, dtype = np.float64)
    q[y] += 1 - eps
    shifted_logits = logits - np.max(logits)
    log_softmax = shifted_logits - np.log(np.sum(np.exp(shifted_logits)))
    loss = -(np.sum(q*log_softmax))
    return (q,loss)
