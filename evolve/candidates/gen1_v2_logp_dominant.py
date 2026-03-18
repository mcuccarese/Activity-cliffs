"""Hypothesis: LogP change is the dominant signal (Hansch pi parameter analog)."""
import numpy as np

def score_transformations(X):
    # LogP is the Hansch pi equivalent — dominant in Topliss
    logp = np.abs(X[:, 1])
    # TPSA captures polar complement
    tpsa = np.abs(X[:, 2])
    # H-bond changes are binary-like and highly impactful
    hbd = np.abs(X[:, 3])
    hba = np.abs(X[:, 4])
    # Nonlinear: squared LogP (big changes disproportionately informative)
    return logp ** 1.5 + tpsa * 0.03 + (hbd + hba) * 0.8
