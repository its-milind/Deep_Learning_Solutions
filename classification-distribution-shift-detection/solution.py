import numpy as np

_EPS = 1e-6


def bin_proportions(values: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    counts,_ = np.histogram(values, bin_edges)
    return counts/len(values)


def population_stability_index(
    train_values: np.ndarray, live_values: np.ndarray, num_bins: int = 10
) -> float:
    
    quantiles = np.linspace(0, 1, num_bins+1)
    bin_edges = np.quantile(train_values, quantiles)

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    
    train_pct = np.clip(bin_proportions(train_values, bin_edges), _EPS, None)
    live_pct = np.clip(bin_proportions(live_values, bin_edges), _EPS, None)

    PSI = np.sum( (live_pct - train_pct) * np.log(live_pct / train_pct) )
    return float(PSI)


def detect_distribution_shift(
    train_values: np.ndarray, live_values: np.ndarray, num_bins: int = 10, threshold: float = 0.2
) -> bool:
    psi_threshold = population_stability_index(train_values, live_values, num_bins)
    return bool(psi_threshold>threshold)
