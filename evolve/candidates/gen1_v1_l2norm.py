"""Hypothesis: L2 norm of delta vector captures overall magnitude of change."""
import numpy as np

def score_transformations(X):
    # L2 norm of standardized delta descriptors
    deltas = X[:, :7]
    # Rough normalisation by typical range
    scales = np.array([100.0, 2.0, 40.0, 2.0, 2.0, 3.0, 7.0], dtype=np.float32)
    normed = deltas / scales
    l2 = np.sqrt(np.sum(normed ** 2, axis=1))
    # Add Tanimoto dissimilarity
    dissim = 1.0 - X[:, 7]
    return l2 + dissim * 0.5
