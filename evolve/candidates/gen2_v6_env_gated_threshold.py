"""Env-gated thresholds: at sensitive positions (high env cliff rate),
even small property changes are informative. At insensitive positions,
only large property changes matter."""
import numpy as np

def score_transformations(X):
    env_sens = X[:, 10]  # env_r2_cliff_rate (0 to ~0.65)

    # Dynamic threshold: at sensitive positions (high env), accept smaller changes
    # At insensitive positions, require larger changes
    threshold = 0.5 * (1.0 - env_sens)  # lower threshold at sensitive positions

    # Total property change magnitude (normalised)
    prop_mag = (
        np.abs(X[:, 1]) / 2.5    # LogP normalised
        + np.abs(X[:, 6]) / 7.0  # HAC normalised
        + np.abs(X[:, 3]) / 2.0  # HBD
        + np.abs(X[:, 4]) / 2.0  # HBA
    ) / 4.0  # average, roughly 0 to 1

    # How much the property change exceeds the threshold
    excess = np.maximum(prop_mag - threshold, 0.0)

    # Final score: environment sensitivity × (1 + excess)
    return env_sens * 3.0 + excess * 2.0 + X[:, 11] * 1.5
