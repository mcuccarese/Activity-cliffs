"""Hypothesis: FP Tanimoto (structural dissimilarity) is the strongest single signal.
Very different R-groups cause bigger SAR jumps. Property deltas are secondary."""
import numpy as np

def score_transformations(X):
    dissim = 1.0 - X[:, 7]  # 0 = identical, 1 = completely different

    # Dissimilarity raised to power < 1 to boost moderate differences
    dissim_score = dissim ** 0.7 * 3.0

    # Small property delta corrections
    logp_bonus = np.abs(X[:, 1]) * 0.3
    hbond_bonus = (np.abs(X[:, 3]) + np.abs(X[:, 4])) * 0.2

    return dissim_score + logp_bonus + hbond_bonus
