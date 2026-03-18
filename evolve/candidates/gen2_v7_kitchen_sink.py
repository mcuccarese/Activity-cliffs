"""Everything: Ridge-optimal weights + env context + interactions + nonlinearity."""
import numpy as np

def score_transformations(X):
    # === Context (dominant signal) ===
    env_r2_cr = X[:, 10]   # cliff rate at r=2 env
    env_r1_cr = X[:, 8]    # cliff rate at r=1 env
    env_r2_md = X[:, 11]   # mean delta at r=2
    env_r1_md = X[:, 9]    # mean delta at r=1

    # === Property deltas (Ridge-optimal, secondary signal) ===
    abs_hac = np.abs(X[:, 6])
    abs_logp = np.abs(X[:, 1])
    abs_hba = np.abs(X[:, 4])
    abs_tpsa = np.abs(X[:, 2])
    abs_hbd = np.abs(X[:, 3])
    abs_rotb = np.abs(X[:, 5])
    dissim = 1.0 - X[:, 7]

    # === Main scoring ===
    # Context features (primary)
    ctx = env_r2_cr * 4.0 + env_r1_cr * 1.5 + env_r2_md * 1.0 + env_r1_md * 0.3

    # Property features (Ridge-learned weights)
    prop = (abs_hac * 0.05 + abs_logp * 0.25 + abs_hba * 0.15
            + abs_tpsa * 0.015 + abs_hbd * 0.1 + abs_rotb * 0.08 + dissim * 0.3)

    # Interaction: context × property (amplification effect)
    interaction = env_r2_cr * (abs_logp * 0.5 + abs_hbd * 0.3 + dissim * 0.2)

    # Nonlinearity: diminishing returns for huge HAC changes
    nonlin = -abs_hac ** 2 * 0.002

    return ctx + prop + interaction + nonlin
