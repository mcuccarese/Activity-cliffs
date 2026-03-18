"""Context + property deltas: env cliff rate modulated by property changes."""
import numpy as np

def score_transformations(X):
    env_score = X[:, 10] * 3.0 + X[:, 8] * 1.0  # context baseline
    delta_score = (
        np.abs(X[:, 6]) * 0.05   # HAC (Ridge top feature)
        + np.abs(X[:, 1]) * 0.3  # LogP
        + np.abs(X[:, 4]) * 0.15 # HBA
        + np.abs(X[:, 2]) * 0.01 # TPSA
        + (1.0 - X[:, 7]) * 0.2  # dissimilarity
    )
    return env_score + delta_score
