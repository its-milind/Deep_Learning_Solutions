def gradient_checkpoint(
    x: float, y: float, w1: float, b1: float, w2: float, b2: float
) -> tuple[float, float, float, float]:
    """
    Two-layer scalar network: h = ReLU(w1*x + b1), yhat = w2*h + b2,
    L = 0.5 * (yhat - y)^2.

    Return (h, yhat, loss, dL_dw1), where dL_dw1 is computed by the chain
    rule, using ReLU'(z) = 1 if z > 0 else 0 on the pre-activation z.
    """
    # TODO: evaluate the ReLU gate on the pre-activation, not on h.
    out = w1*x + b1
    h = out if out>0 else 0
    y_hat = w2*h + b2
    L = 0.5 * ( (y_hat-y)**2 )
    dL_dW1 = (y_hat-y) * w2 * ( 1 if h else 0) * x
    return ( h, y_hat, L, dL_dW1)
