import numpy as np


def temperature_calibrator(z: np.ndarray, tau: float) -> np.ndarray:
    """
    z: logits.
    tau: temperature, tau > 0.

    p = softmax(z / tau), using an overflow-safe softmax (subtract the
    max before exponentiating). Divide by tau, never multiply.

    Return p.
    """
    # TODO: scale by dividing, then stable softmax.
    z = z - np.max(z)
    z = z/tau
    exponents = np.exp(z)
    p = exponents / np.sum(exponents)
    return p
