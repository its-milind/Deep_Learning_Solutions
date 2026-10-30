import numpy as np


def mse_gradient(
    input: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    input:  shape (batch_size, in_features)
    weight: shape (out_features, in_features)
    bias:   shape (out_features,), or None
    target: shape (batch_size, out_features)

    Returns:
        grad_weight: same shape as weight
        grad_bias: same shape as bias, or None if bias is None
    """
    # TODO: Implement the gradient formulas derived in Theory.
    # Do not use an autograd library -- these are the manual derivatives.
    b = bias if bias is not None else 0.0
    prediction = input@weight.T + b
    loss = (target - prediction)**2
    N = target.size
    dldp = (2/N)*(prediction - target)
    
    grad_bias = np.sum(dldp,axis = 0) if bias is not None else None
    grad_weight = (dldp.T) @ input 
    return [grad_weight, grad_bias]
