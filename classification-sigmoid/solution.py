import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    z: any shape of real numbers.
    Returns elementwise sigmoid, same shape as z, values strictly in (0, 1).
    """
    # TODO: Implement the sigmoid function from Theory.
    # Clip z before exponentiating so large |z| never overflows or
    # returns nan.
    
    new_z = np.clip(z,-500,500)
    new_z = 1/(1+ np.exp(new_z*-1) )
    return new_z
