
import numpy as np

from _load import load_solution

linear = load_solution("linear-regression-hypothesis-function").linear
sigmoid = load_solution("classification-sigmoid").sigmoid
train_logistic_regression = load_solution("classification-training-loop").train_logistic_regression


def train_one_vs_rest(
    input: np.ndarray, target: np.ndarray, num_classes: int, lr: float, epochs: int
) -> tuple[np.ndarray, np.ndarray]:
    
    _, features = input.shape
    weights = np.zeros((num_classes, features))
    bias = np.zeros((num_classes,))

    for k in range(num_classes):
        binary_target = (target == k).astype(input.dtype)
        w_k, b_k = train_logistic_regression(input, binary_target, lr = lr, epochs = epochs)
        weights[k] = np.squeeze(w_k)
        bias[k] = np.squeeze(b_k)
    return weights, bias
        
    


def predict_one_vs_rest(input: np.ndarray, weights: np.ndarray, biases: np.ndarray) -> np.ndarray:
    logits = linear(input, weights, biases)
    probs = sigmoid(logits)
    return np.argmax(probs, axis = -1)
