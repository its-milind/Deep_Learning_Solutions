import numpy as np


def train_linear_regression(
    input: np.ndarray,
    target: np.ndarray,
    lr: float,
    epochs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    input:  shape (batch_size, in_features)
    target: shape (batch_size,), one target value per sample
    lr: learning rate
    epochs: number of full-batch gradient-descent steps

    Returns:
        weight: shape (1, in_features)
        bias: shape (1,)
    """
    # TODO: Initialize weight to zeros (1, in_features), bias to zeros (1,).
    # Reshape target to (batch_size, 1) once, up front.
    # Repeat for `epochs` iterations:
    #   1. Compute gradients with mse_gradient()
    #   2. Update weight, bias with gd_step()
    # Reuse those functions -- don't reimplement their logic here.
    def mse_gradient(predictions,loss, input, target):
        N = target.size
        dldp = (2/N)* (predictions-target)
        grad_weight = dldp.T @ input 
        grad_bias = np.sum(dldp, axis = 0)
        return (grad_weight, grad_bias)
        
    def gd_step(grad_weight, grad_bias, weights, bias, lr):
        weights = weights - lr*grad_weight
        bias = bias - lr*grad_bias
        return (weights,bias)
    batch_size, input_features = input.shape
    weights = np.zeros((1, input_features))
    bias = np.zeros(1,)
    target = target.reshape(-1,1)
    for epoch in range(epochs):
        predictions = input@weights.T + bias
        sq_error = (target-predictions)**2
        loss = np.mean(sq_error)
       
        grad_weight, grad_bias = mse_gradient(predictions, loss, input, target)
        weights, bias = gd_step(grad_weight, grad_bias, weights, bias, lr)
    return (weights, bias)
