
import numpy as np

from _load import load_solution


def train_val_split(
    input: np.ndarray, target: np.ndarray, val_fraction: float = 0.2, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Shuffles `input`/`target` together (so a row's features and its
    target stay paired), then splits off `val_fraction` of the rows as
    a held-out validation set.

    Returns (input_train, target_train, input_val, target_val).
    """
    if seed is not None:
        rng = np.random.default_rng(seed)

    n = input.shape[0]
    val_pct = round(n*val_fraction)

    indexes = rng.permutation(n)
    
    train_idx = indexes[:-val_pct]
    val_idx = indexes[-val_pct:]

    input_train, target_train = input[train_idx], target[train_idx]
    input_val, target_val = input[val_idx], target[val_idx]
    
    return (input_train, target_train, input_val, target_val)
    

def generalization_gap(train_loss: float, val_loss: float) -> float:
    """
    The generalization gap: how much worse the model performs on data
    it never trained on, compared to the data it did. A large positive
    gap is the numerical signature of overfitting.
    """
    return val_loss - train_loss
