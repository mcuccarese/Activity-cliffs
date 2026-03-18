"""Hypothesis: Direction matters — increasing LogP is different from decreasing it.
In kinase SAR, adding lipophilicity often boosts potency (hydrophobic pocket filling)."""
import numpy as np

def score_transformations(X):
    dlogp = X[:, 1]  # SIGNED
    dtpsa = X[:, 2]  # SIGNED
    dhbd = X[:, 3]
    dhba = X[:, 4]

    # Increasing LogP (adding lipophilicity) is informative in different way than decreasing
    # Use asymmetric weighting
    logp_score = np.where(dlogp > 0, dlogp * 1.2, np.abs(dlogp) * 0.8)

    # Decreasing TPSA (removing polarity) often accompanies potency increase
    tpsa_score = np.where(dtpsa < 0, np.abs(dtpsa) * 0.04, np.abs(dtpsa) * 0.02)

    # Removing H-bond donors is often more dramatic than adding
    hbd_score = np.where(dhbd < 0, np.abs(dhbd) * 0.8, np.abs(dhbd) * 0.4)
    hba_score = np.abs(dhba) * 0.4

    # Overall magnitude still matters
    mw_score = np.abs(X[:, 0]) * 0.008
    dissim = (1.0 - X[:, 7]) * 0.4

    return logp_score + tpsa_score + hbd_score + hba_score + mw_score + dissim
