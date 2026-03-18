"""Multiplicative: env context AMPLIFIES property changes.
If you're at a sensitive position AND make a big property change → cliff."""
import numpy as np

def score_transformations(X):
    # Context sensitivity (0 to ~0.5)
    env_sensitivity = X[:, 10] * 0.6 + X[:, 8] * 0.4

    # Magnitude of property change (higher = bigger change)
    prop_change = (
        np.abs(X[:, 1]) * 0.4    # LogP
        + np.abs(X[:, 6]) * 0.05 # HAC
        + np.abs(X[:, 3]) * 0.2  # HBD
        + np.abs(X[:, 4]) * 0.2  # HBA
        + (1.0 - X[:, 7]) * 0.3  # dissimilarity
    )

    # Multiplicative: sensitive position × big change
    return env_sensitivity * prop_change + env_sensitivity * 2.0 + prop_change * 0.3
