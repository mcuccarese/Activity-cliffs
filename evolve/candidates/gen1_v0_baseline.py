"""Baseline: weighted absolute deltas (from initial.py). NDCG@5 ≈ 0.513"""
import numpy as np

def score_transformations(X):
    return (
        np.abs(X[:, 0]) * 0.01
        + np.abs(X[:, 1]) * 1.0
        + np.abs(X[:, 2]) * 0.02
        + np.abs(X[:, 3]) * 0.5
        + np.abs(X[:, 4]) * 0.5
        + np.abs(X[:, 5]) * 0.3
        + np.abs(X[:, 6]) * 0.1
        + (1.0 - X[:, 7]) * 0.5
    )
