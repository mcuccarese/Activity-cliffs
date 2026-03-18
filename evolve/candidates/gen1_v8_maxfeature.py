"""Hypothesis: A single dominant property change drives cliffs.
The MAX of any normalized delta is more predictive than the SUM."""
import numpy as np

def score_transformations(X):
    # Normalize each delta to [0, ~1] range
    scales = np.array([100.0, 2.5, 40.0, 2.0, 2.0, 3.0, 7.0], dtype=np.float32)
    normed = np.abs(X[:, :7]) / scales

    # Max across features: which single property changed the most?
    max_change = np.max(normed, axis=1)

    # Also count how many features changed significantly (>0.3 of range)
    num_big = np.sum(normed > 0.3, axis=1).astype(np.float64)

    # Dissimilarity
    dissim = 1.0 - X[:, 7]

    return max_change * 2.0 + num_big * 0.3 + dissim * 0.5
