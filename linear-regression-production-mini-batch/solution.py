import numpy as np


def train_linear_regression_production(
    input: np.ndarray,
    target: np.ndarray,
    lr: float,
    epochs: int,
    batch_size: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mini-batch gradient descent, reusing mse_gradient / gd_step per batch
    instead of once per epoch over the full dataset.

    seed: pass a fixed integer to make training reproducible run to run.
    Leave as None for a genuinely random shuffle order.
    """
    # TODO: Each epoch, shuffle the sample order, split into batches of
    # batch_size, and take one gradient step per batch (not per epoch).
    # The last batch may be smaller -- handle that without crashing.
    # Build the RNG (np.random.default_rng(seed)) ONCE, before the epoch
    # loop -- not inside it. Re-creating it every epoch would give every
    # epoch the identical shuffle order.
    def gd_step(weights, bias,  lr, grad_weight, grad_bias):
        weights = weights - lr*grad_weight
        bias = bias - lr*grad_bias
        return weights, bias
        
    rng = np.random.default_rng(seed)

    samples, features = input.shape
    weights = np.zeros((1, features))
    bias = np.zeros((1,))
    
    indices = np.arange(samples)                                #initializing indexes to shuffle
    target = target.reshape(samples,1)
    
    for epoch in range(epochs):
        rng.shuffle(indices)
        for batch in range(0,samples, batch_size):
            batch_idx = indices[batch : batch+batch_size]       #extracting indexes in a batch       
            X_train = input[batch_idx]
            y_train = target[batch_idx]
            
            pred = X_train@weights.T + bias                      
            loss = (y_train - pred)**2
            N = X_train.shape[0]                               #samples in the batch
            dldp = (2.0/N) *(pred - y_train)
            grad_weight = dldp.T@X_train 
            grad_bias = np.sum(dldp, axis = 0)

            weights, bias = gd_step(weights, bias,lr,          #weight and bias updation through 
                                    grad_weight, grad_bias)    #gardient descent
            
    return (weights, bias)
