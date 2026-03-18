"""Hypothesis: Cliff-inducing changes have a 'sweet spot' — too small = no effect,
too large = comparing apples to oranges (different binding modes entirely).
Sigmoid gating to focus on moderate-to-large changes."""
import numpy as np

def score_transformations(X):
    def soft_threshold(x, center, width):
        """Sigmoid: rises around center, width controls steepness."""
        return 1.0 / (1.0 + np.exp(-(np.abs(x) - center) / width))

    # LogP changes >0.5 start being informative, plateau around 2.0
    logp_gate = soft_threshold(X[:, 1], center=0.5, width=0.3)
    # TPSA changes >10 Å² are meaningful
    tpsa_gate = soft_threshold(X[:, 2], center=10.0, width=5.0)
    # HBD/HBA: any change ≥1 is significant
    hbd_gate = soft_threshold(X[:, 3], center=0.5, width=0.3)
    hba_gate = soft_threshold(X[:, 4], center=0.5, width=0.3)
    # Heavy atom count >3 difference
    hac_gate = soft_threshold(X[:, 6], center=3.0, width=1.5)

    score = (
        logp_gate * 1.5
        + tpsa_gate * 0.8
        + hbd_gate * 1.0
        + hba_gate * 0.8
        + hac_gate * 0.3
        + (1.0 - X[:, 7]) * 0.6  # dissimilarity
    )
    return score
