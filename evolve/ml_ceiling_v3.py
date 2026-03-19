"""
M6c: ML ceiling with 3D context × change-type interaction features.

Tests whether the interaction features break the 0.52 NDCG@5 ceiling
that was the limit of 2D-only descriptors.

Feature sets tested:
  A. Original 8 (abs deltas + dissim)           — baseline (0.517)
  B. 3D context only (9)                        — position-level signal
  C. Change-type deltas only (11)               — within-group variance
  D. Context + change-type (20)                 — additive, no interactions
  E. Interaction features only (99)             — cross-products
  F. Original 8 + change-type (19)              — 2D + within-group
  G. Original 8 + context + change-type (28)    — all features, no interactions
  H. Original 8 + interactions (107)            — 2D + cross-products
  I. ALL features (127)                         — kitchen sink
  J. Original 8 + change-type + interactions (118) — 2D + change-type + cross

All with leave-one-target-out HGB evaluation.

Usage:
    python evolve/ml_ceiling_v3.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

sys.stdout.reconfigure(line_buffering=True)

# ── Load v3 eval data ─────────────────────────────────────────────────────────

EVAL_DATA_PATH = Path(__file__).parent / "eval_data" / "eval_data_v3.npz"
_data = np.load(EVAL_DATA_PATH, allow_pickle=True)
X = _data["X"]
Y = _data["y"]
GROUPS = _data["groups"]
OFFSETS = _data["target_offsets"]
TARGETS = _data["target_names"]
FEAT_NAMES = list(_data["feature_names"])
del _data

print(f"Loaded: {X.shape[0]:,} rows, {X.shape[1]} features, {len(TARGETS)} targets")

# ── Identify feature slices ───────────────────────────────────────────────────
# Layout: [8 deltas+tani | 9 ctx | 11 change_type | 99 interactions]
N_DELTA = 8
N_CTX = 9
N_CT = 11
N_INTERACT = N_CTX * N_CT  # 99

SL_DELTA = slice(0, N_DELTA)
SL_CTX = slice(N_DELTA, N_DELTA + N_CTX)
SL_CT = slice(N_DELTA + N_CTX, N_DELTA + N_CTX + N_CT)
SL_INTERACT = slice(N_DELTA + N_CTX + N_CT, N_DELTA + N_CTX + N_CT + N_INTERACT)

print(f"  Deltas:       cols {SL_DELTA.start}-{SL_DELTA.stop - 1}")
print(f"  3D context:   cols {SL_CTX.start}-{SL_CTX.stop - 1}")
print(f"  Change-type:  cols {SL_CT.start}-{SL_CT.stop - 1}")
print(f"  Interactions: cols {SL_INTERACT.start}-{SL_INTERACT.stop - 1}")


# ── Evaluation functions ──────────────────────────────────────────────────────

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

    print(f"  {label:55s} NDCG@5 = {mean_ndcg:.4f}  (median {median_ndcg:.4f})  [{elapsed:.0f}s]")
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
    print(f"  {label:55s} NDCG@5 = {mean_ndcg:.4f}")
    return mean_ndcg, target_ndcgs


def main():
    print("\n" + "=" * 78)
    print("M6c: ML Ceiling with 3D Context × Change-Type Interactions")
    print("=" * 78)

    # ── Baselines (no ML) ─────────────────────────────────────────────────────
    print("\n--- Baselines (no training) ---")
    score_with_function(lambda _: np.random.RandomState(42).randn(len(Y)), "Random")

    def l2_norm_fn(x):
        deltas = x[:, :7]
        scales = np.array([100.0, 2.0, 40.0, 2.0, 2.0, 3.0, 7.0], dtype=np.float32)
        normed = deltas / scales
        l2 = np.sqrt(np.sum(normed ** 2, axis=1))
        return l2 + (1.0 - x[:, 7]) * 0.5
    score_with_function(l2_norm_fn, "L2 norm (best manual, v1 ceiling)")

    # ── HGB with different feature sets ───────────────────────────────────────
    print("\n--- HistGradientBoosting (leave-one-target-out, k=5) ---")

    # A. Baseline: original 8 (abs deltas + dissimilarity)
    X_orig = np.column_stack([np.abs(X[:, :7]), 1.0 - X[:, 7]])
    leave_one_target_out(X_orig, Y, label="A. Original 8 (abs deltas + dissim)")

    # B. 3D context only
    X_ctx = X[:, SL_CTX]
    leave_one_target_out(X_ctx, Y, label="B. 3D context only (9)")

    # C. Change-type deltas only
    X_ct = X[:, SL_CT]
    leave_one_target_out(X_ct, Y, label="C. Change-type deltas only (11)")

    # D. Context + change-type (no interactions)
    X_ctx_ct = np.column_stack([X[:, SL_CTX], X[:, SL_CT]])
    leave_one_target_out(X_ctx_ct, Y, label="D. Context + change-type (20)")

    # E. Interaction features only
    X_int = X[:, SL_INTERACT]
    leave_one_target_out(X_int, Y, label="E. Interactions only (99)")

    # F. Original 8 + change-type
    X_orig_ct = np.column_stack([X_orig, X[:, SL_CT]])
    leave_one_target_out(X_orig_ct, Y, label="F. Original 8 + change-type (19)")

    # G. Original 8 + context + change-type (no interactions)
    X_orig_ctx_ct = np.column_stack([X_orig, X[:, SL_CTX], X[:, SL_CT]])
    leave_one_target_out(X_orig_ctx_ct, Y, label="G. Original 8 + ctx + change-type (28)")

    # H. Original 8 + interactions
    X_orig_int = np.column_stack([X_orig, X[:, SL_INTERACT]])
    leave_one_target_out(X_orig_int, Y, label="H. Original 8 + interactions (107)")

    # I. ALL features
    X_all = np.column_stack([X_orig, X[:, SL_CTX], X[:, SL_CT], X[:, SL_INTERACT]])
    leave_one_target_out(X_all, Y, label="I. ALL features (127)")

    # J. Original 8 + change-type + interactions (skip raw context)
    X_orig_ct_int = np.column_stack([X_orig, X[:, SL_CT], X[:, SL_INTERACT]])
    leave_one_target_out(X_orig_ct_int, Y, label="J. Original 8 + change-type + interact (118)")

    # ── Deeper HGB on best combo ──────────────────────────────────────────────
    print("\n--- Deeper HGB on promising combos ---")
    deep_kwargs = {
        "max_iter": 500,
        "max_depth": 8,
        "learning_rate": 0.05,
        "min_samples_leaf": 30,
        "random_state": 42,
    }
    leave_one_target_out(X_all, Y, model_kwargs=deep_kwargs,
                         label="I-deep. ALL features, deeper HGB (127)")
    leave_one_target_out(X_orig_ct_int, Y, model_kwargs=deep_kwargs,
                         label="J-deep. Orig + CT + interact, deeper HGB (118)")

    print("\n" + "=" * 78)
    print("Done. If any combo > 0.52, the interaction features break the ceiling!")
    print("=" * 78)


if __name__ == "__main__":
    main()
