"""Direct prediction: use env_mean_delta as the primary score.
The historical mean |delta| at this position IS the expected informativeness."""
import numpy as np

def score_transformations(X):
    # env_r2_mean_delta is literally "how much activity changes at this position on average"
    # This should be extremely predictive
    score = X[:, 11] * 2.0 + X[:, 9] * 1.0  # weighted mean deltas

    # Small bonus for larger property changes
    score += np.abs(X[:, 1]) * 0.1  # LogP
    score += (1.0 - X[:, 7]) * 0.05  # dissimilarity

    return score
