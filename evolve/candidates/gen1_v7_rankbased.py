"""Hypothesis: Use rank-transformed features to be robust to outliers.
Percentile-based scoring avoids sensitivity to extreme deltas."""
import numpy as np

def score_transformations(X):
    # Rank-transform each feature within the evaluation set
    N = X.shape[0]
    scores = np.zeros(N, dtype=np.float64)
    weights = [0.01, 1.0, 0.02, 0.5, 0.5, 0.2, 0.1]

    for j in range(7):
        absvals = np.abs(X[:, j])
        # Rank percentile (0 to 1)
        order = np.argsort(absvals)
        ranks = np.empty(N, dtype=np.float64)
        ranks[order] = np.linspace(0, 1, N)
        scores += ranks * weights[j]

    # Tanimoto dissimilarity rank
    dissim = 1.0 - X[:, 7]
    order = np.argsort(dissim)
    ranks = np.empty(N, dtype=np.float64)
    ranks[order] = np.linspace(0, 1, N)
    scores += ranks * 0.5

    return scores
