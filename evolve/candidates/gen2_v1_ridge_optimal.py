"""Ridge-optimal linear weights from OLS regression on abs_delta_pActivity.
Uses absolute deltas + dissimilarity + key interactions."""
import numpy as np

def score_transformations(X):
    absX = np.abs(X[:, :7])
    dissim = 1.0 - X[:, 7]

    # Ridge-learned weights (top features)
    score = (
        absX[:, 6] * 0.112        # |delta_HAC| — strongest single feature
        + absX[:, 1] * 0.062      # |delta_LogP|
        + absX[:, 4] * 0.040      # |delta_HBA|
        + absX[:, 2] * 0.037      # |delta_TPSA|
        + absX[:, 5] * 0.029      # |delta_RotBonds|
        + absX[:, 3] * 0.024      # |delta_HBD|
        + absX[:, 0] * 0.01       # |delta_MW| (collinear with HAC)
        + dissim * 0.089           # FP dissimilarity
    )
    # Nonlinear: squared HAC and MW (diminishing returns for huge changes)
    score += X[:, 0] ** 2 * 3.5e-5   # MW²
    score -= absX[:, 6] ** 2 * 0.006  # HAC² (negative = diminishing returns)

    return score
