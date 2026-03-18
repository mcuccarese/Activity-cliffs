"""
Template scoring function for MMP transformation ranking.

Evolves via ShinkaEvolve to discover interpretable rules for predicting which
R-group modifications cause the largest activity changes — a data-driven
generalized Topliss tree for medicinal chemistry SAR exploration.
"""
import numpy as np


# EVOLVE-BLOCK-START
def score_transformations(X: np.ndarray) -> np.ndarray:
    """
    Score how informative each R-group swap is for revealing steep SAR.

    Parameters
    ----------
    X : float32 array of shape (N, 12), columns:
        --- Property deltas (signed) ---
        0: delta_MW              - molecular weight change (Daltons)
        1: delta_LogP            - lipophilicity change
        2: delta_TPSA            - polar surface area change (A^2)
        3: delta_HBDonors        - H-bond donor count change
        4: delta_HBAcceptors     - H-bond acceptor count change
        5: delta_RotBonds        - rotatable bond count change
        6: delta_HeavyAtomCount  - heavy atom count change

        --- Structural similarity ---
        7: fp_tanimoto           - R-group fingerprint Tanimoto (0=different, 1=same)

        --- Attachment environment context ---
        8: env_r1_cliff_rate     - historical cliff rate at this attachment env (radius 1)
        9: env_r1_mean_delta     - historical mean |delta_pActivity| at this env (r1)
       10: env_r2_cliff_rate     - historical cliff rate at this env (radius 2, more specific)
       11: env_r2_mean_delta     - historical mean |delta_pActivity| at this env (r2)

    Returns
    -------
    scores : float64 array of shape (N,)
        Higher = more likely to cause a large activity change (cliff).

    Scientific context
    ------------------
    This is a generalized Topliss tree.  The env features capture WHERE on the
    molecule the change happens (e.g., a nitrogen-adjacent aromatic position has
    different SAR sensitivity than a terminal alkyl position).  The delta features
    capture WHAT changes (physicochemical property shifts).

    Key principles:
    - Attachment environment is the strongest predictor of cliff probability
    - Large |delta_LogP| indicates hydrophobic pocket filling / disruption
    - H-bond donor/acceptor changes affect binding site interactions
    - Low fp_tanimoto = structurally different R-groups = bigger SAR jump
    - Nonlinear interactions between context and property changes matter

    Available: numpy, scipy, sklearn (helpers only). Must be vectorised.
    """
    # Baseline: context features + weighted property deltas
    score = (
        X[:, 10] * 3.0            # env_r2_cliff_rate — strongest single signal
        + X[:, 8] * 1.0           # env_r1_cliff_rate
        + np.abs(X[:, 6]) * 0.05  # |delta_HeavyAtomCount|
        + np.abs(X[:, 1]) * 0.3   # |delta_LogP|
        + (1.0 - X[:, 7]) * 0.2   # FP dissimilarity
    )
    return score
# EVOLVE-BLOCK-END
