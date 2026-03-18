"""Pure context: only env features. How far can context alone take us?"""
import numpy as np

def score_transformations(X):
    # Use only attachment environment features
    # r2 is more specific (3108 unique) so should be more informative
    return X[:, 10] * 2.0 + X[:, 8] * 1.0 + X[:, 11] * 0.5 + X[:, 9] * 0.3
