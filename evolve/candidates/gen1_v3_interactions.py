"""Hypothesis: Feature interactions matter — LogP×TPSA captures amphiphilic changes."""
import numpy as np

def score_transformations(X):
    dlogp = np.abs(X[:, 1])
    dtpsa = np.abs(X[:, 2])
    dhbd = np.abs(X[:, 3])
    dhba = np.abs(X[:, 4])
    dmw = np.abs(X[:, 0])
    dissim = 1.0 - X[:, 7]

    # Main effects
    score = dlogp * 1.0 + dtpsa * 0.02 + (dhbd + dhba) * 0.5

    # Interaction: simultaneous LogP and TPSA change = amphiphilic shift
    score += dlogp * dtpsa * 0.02

    # Interaction: large MW change with LogP = adding hydrophobic bulk
    score += dmw * dlogp * 0.005

    # Dissimilar R-groups bonus
    score += dissim * 0.3

    return score
