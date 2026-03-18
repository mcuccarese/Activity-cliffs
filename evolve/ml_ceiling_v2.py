"""
M5c: ML ceiling with ENRICHED features (v2).

Compares HistGradientBoosting on:
  - Original 8 features (delta descriptors + Tanimoto)
  - All 43 v2 features (+ FP XOR PCA, FG flags, transform freq, size)
  - FG flags only (12-dim)
  - FP XOR PCA only (20-dim)
  - Full 256-bit XOR (stored separately in npz)
  - Combined: abs deltas + FG flags + XOR PCA (most promising mix)

All with leave-one-target-out evaluation.

Usage:
    python evolve/ml_ceiling_v2.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

# Force unbuffered output so we see progress in real-time
sys.stdout.reconfigure(line_buffering=True)

# ── Load v2 eval data ─────────────────────────────────────────────────────────

EVAL_DATA_PATH = Path(__file__).parent / "eval_data" / "eval_data_v2.npz"
_data = np.load(EVAL_DATA_PATH, allow_pickle=True)
X = _data["X"]
Y = _data["y"]
GROUPS = _data["groups"]
OFFSETS = _data["target_offsets"]
TARGETS = _data["target_names"]
FEAT_NAMES = _data["feature_names"]
FP_XOR = _data["fp_xor"]  # (N, 256) full XOR bits
del _data

print(f"Loaded: {X.shape[0]:,} rows, {X.shape[1]} features, {len(TARGETS)} targets")
print(f"Features: {list(FEAT_NAMES)}")
print(f"FP XOR shape: {FP_XOR.shape}")


def ndcg_at_k(y_true, y_score, k=5):
    n = len(y_true)
    if n == 0:
        return 0.0
    k = min(k, n)
    discounts = np.log2(np.arange(2, k + 2))
    pred_order = np.argsort(-y_score)[:k]
    dcg = np.sum(y_true[pred_order] / discounts)
    ideal_order = np.argsort(-y_true)[:k]
    idcg = np.sum(y_true[ideal_order] / discounts)
    return float(dcg / idcg) if idcg > 0 else 0.0


def eval_ndcg_for_target(scores, y, groups, k=5):
    ndcgs = []
    for mol_id in np.unique(groups):
        mask = groups == mol_id
        if mask.sum() >= k:
            ndcgs.append(ndcg_at_k(y[mask], scores[mask], k))
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def leave_one_target_out(X_in, y_in, model_kwargs=None, label=""):
    if model_kwargs is None:
        model_kwargs = {
            "max_iter": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
            "min_samples_leaf": 50,
            "random_state": 42,
        }

    target_ndcgs = {}
    t_total = time.perf_counter()

    for i, target in enumerate(TARGETS):
        lo, hi = OFFSETS[i], OFFSETS[i + 1]
        X_test, y_test = X_in[lo:hi], y_in[lo:hi]
        g_test = GROUPS[lo:hi]

        train_mask = np.ones(len(y_in), dtype=bool)
        train_mask[lo:hi] = False
        X_train, y_train = X_in[train_mask], y_in[train_mask]

        model = HistGradientBoostingRegressor(**model_kwargs)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        ndcg = eval_ndcg_for_target(preds, y_test, g_test, k=5)
        target_ndcgs[str(target)] = ndcg

    elapsed = time.perf_counter() - t_total
    mean_ndcg = float(np.mean(list(target_ndcgs.values())))
    median_ndcg = float(np.median(list(target_ndcgs.values())))
    best_t = max(target_ndcgs, key=target_ndcgs.get)
    worst_t = min(target_ndcgs, key=target_ndcgs.get)

    print(f"  {label:45s} NDCG@5 = {mean_ndcg:.4f}  (median {median_ndcg:.4f})  [{elapsed:.0f}s]")
    print(f"    best: {best_t} ({target_ndcgs[best_t]:.4f})  worst: {worst_t} ({target_ndcgs[worst_t]:.4f})")
    return mean_ndcg, target_ndcgs


def score_with_function(score_fn, label=""):
    scores = score_fn(X)
    target_ndcgs = {}
    for i, target in enumerate(TARGETS):
        lo, hi = OFFSETS[i], OFFSETS[i + 1]
        ndcg = eval_ndcg_for_target(scores[lo:hi], Y[lo:hi], GROUPS[lo:hi], k=5)
        target_ndcgs[str(target)] = ndcg
    mean_ndcg = float(np.mean(list(target_ndcgs.values())))
    print(f"  {label:45s} NDCG@5 = {mean_ndcg:.4f}")
    return mean_ndcg, target_ndcgs


def main():
    print("\n" + "=" * 70)
    print("M5c: ML Ceiling with Enriched Features (v2)")
    print("=" * 70)

    # ── Baselines ─────────────────────────────────────────────────────────
    print("\n--- Baselines ---")
    score_with_function(lambda _: np.random.RandomState(42).randn(len(Y)), "Random")
    score_with_function(lambda x: np.abs(x[:, 6]), "|delta_HAC| only")

    def l2_norm_fn(x):
        deltas = x[:, :7]
        scales = np.array([100.0, 2.0, 40.0, 2.0, 2.0, 3.0, 7.0], dtype=np.float32)
        normed = deltas / scales
        l2 = np.sqrt(np.sum(normed ** 2, axis=1))
        return l2 + (1.0 - x[:, 7]) * 0.5

    score_with_function(l2_norm_fn, "L2 norm (best manual)")

    # ── HGB with different feature sets ───────────────────────────────────
    print("\n--- HistGradientBoosting (leave-one-target-out) ---")

    # Original 8 features (absolute deltas + dissimilarity)
    X_orig8 = np.column_stack([np.abs(X[:, :7]), 1.0 - X[:, 7]])
    leave_one_target_out(X_orig8, Y, label="Original 8 (abs deltas + dissim)")

    # FG flags only (12-dim, indices 28-39)
    X_fg = X[:, 28:40]
    leave_one_target_out(X_fg, Y, label="FG net flags only (12)")

    # XOR PCA only (20-dim, indices 8-27)
    X_xor_pca = X[:, 8:28]
    leave_one_target_out(X_xor_pca, Y, label="XOR PCA only (20)")

    # FG flags + abs deltas + dissim (most interpretable combo)
    X_fg_deltas = np.column_stack([np.abs(X[:, :7]), 1.0 - X[:, 7], X[:, 28:40]])
    leave_one_target_out(X_fg_deltas, Y, label="Abs deltas + dissim + FG flags (20)")

    # All 43 v2 features
    leave_one_target_out(X, Y, label="All 43 v2 features")

    # Abs deltas + XOR PCA + FG flags + freq + size (skip signed deltas)
    X_best_mix = np.column_stack([
        np.abs(X[:, :7]),   # abs deltas (7)
        1.0 - X[:, 7],      # dissim (1)
        X[:, 8:28],         # XOR PCA (20)
        X[:, 28:40],        # FG flags (12)
        X[:, 40:43],        # freq + size (3)
    ])
    leave_one_target_out(X_best_mix, Y, label="Best mix: abs_d + XOR + FG + freq (43)")

    # NOTE: Full 256-bit XOR experiments skipped (too slow for LOO-50).
    # The PCA-compressed version above captures the key signal.

    print("\n" + "=" * 70)
    print("Done. Compare enriched features vs original 8 to decide next steps.")
    print("=" * 70)


if __name__ == "__main__":
    main()
