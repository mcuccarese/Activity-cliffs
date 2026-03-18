"""
M5c: Measure ML ceiling with HistGradientBoosting on current features.

Leave-one-target-out evaluation: for each target, train a gradient boosting
model on the other 49 targets to predict abs_delta_pActivity, then use the
predictions as scores for NDCG@5 computation.

Also tests:
- Pointwise regression (HistGradientBoosting) on 8 original features
- Pointwise regression on 12 features (with env context)
- Absolute-value augmented features (|delta| + delta)
- Random baseline for sanity check

Usage:
    python evolve/ml_ceiling.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

# ── Load eval data ────────────────────────────────────────────────────────────

EVAL_DATA_PATH = Path(__file__).parent / "eval_data" / "eval_data.npz"
_data = np.load(EVAL_DATA_PATH, allow_pickle=True)
X = _data["X"]
Y = _data["y"]
GROUPS = _data["groups"]
OFFSETS = _data["target_offsets"]
TARGETS = _data["target_names"]
del _data

print(f"Loaded: {X.shape[0]:,} rows, {X.shape[1]} features, {len(TARGETS)} targets")


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
    """Mean NDCG@k across all mol_from groups for one target slice."""
    ndcgs = []
    for mol_id in np.unique(groups):
        mask = groups == mol_id
        if mask.sum() >= k:
            ndcgs.append(ndcg_at_k(y[mask], scores[mask], k))
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def leave_one_target_out_ndcg(X_all, y_all, feature_cols=None, model_kwargs=None):
    """
    Train HistGradientBoosting with leave-one-target-out, return per-target NDCG.
    """
    if model_kwargs is None:
        model_kwargs = {
            "max_iter": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "min_samples_leaf": 50,
            "random_state": 42,
        }

    target_ndcgs = {}
    total_time = 0.0

    for i, target in enumerate(TARGETS):
        lo, hi = OFFSETS[i], OFFSETS[i + 1]

        # Test data: this target
        X_test = X_all[lo:hi] if feature_cols is None else X_all[lo:hi, feature_cols]
        y_test = y_all[lo:hi]
        g_test = GROUPS[lo:hi]

        # Train data: all other targets
        train_mask = np.ones(len(y_all), dtype=bool)
        train_mask[lo:hi] = False
        X_train = X_all[train_mask] if feature_cols is None else X_all[train_mask][:, feature_cols]
        y_train = y_all[train_mask]

        t0 = time.perf_counter()
        model = HistGradientBoostingRegressor(**model_kwargs)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        dt = time.perf_counter() - t0
        total_time += dt

        ndcg = eval_ndcg_for_target(preds, y_test, g_test, k=5)
        target_ndcgs[str(target)] = ndcg

    mean_ndcg = float(np.mean(list(target_ndcgs.values())))
    return mean_ndcg, target_ndcgs, total_time


def score_with_function(score_fn, X_all):
    """Score all rows with a hand-crafted function, compute per-target NDCG."""
    scores = score_fn(X_all)
    target_ndcgs = {}
    for i, target in enumerate(TARGETS):
        lo, hi = OFFSETS[i], OFFSETS[i + 1]
        ndcg = eval_ndcg_for_target(scores[lo:hi], Y[lo:hi], GROUPS[lo:hi], k=5)
        target_ndcgs[str(target)] = ndcg
    mean_ndcg = float(np.mean(list(target_ndcgs.values())))
    return mean_ndcg, target_ndcgs


def main():
    print("\n" + "=" * 70)
    print("M5c: ML Ceiling Measurement")
    print("=" * 70)

    # ── Baseline: random scores ──────────────────────────────────────────
    print("\n--- Random baseline ---")
    rng = np.random.RandomState(42)
    random_scores = rng.randn(len(Y))
    random_ndcg, _ = score_with_function(lambda _: random_scores, X)
    print(f"  Random NDCG@5: {random_ndcg:.4f}")

    # ── Baseline: |delta_HAC| alone ──────────────────────────────────────
    print("\n--- |delta_HAC| baseline ---")
    hac_ndcg, _ = score_with_function(lambda x: np.abs(x[:, 6]), X)
    print(f"  |delta_HAC| NDCG@5: {hac_ndcg:.4f}")

    # ── Best manual candidate: L2 norm ───────────────────────────────────
    print("\n--- Best manual candidate (L2 norm) ---")

    def l2_norm_fn(x):
        deltas = x[:, :7]
        scales = np.array([100.0, 2.0, 40.0, 2.0, 2.0, 3.0, 7.0], dtype=np.float32)
        normed = deltas / scales
        l2 = np.sqrt(np.sum(normed ** 2, axis=1))
        dissim = 1.0 - x[:, 7]
        return l2 + dissim * 0.5

    l2_ndcg, _ = score_with_function(l2_norm_fn, X)
    print(f"  L2 norm NDCG@5: {l2_ndcg:.4f}")

    # ── HistGradientBoosting on 8 original features (LOO-target) ─────────
    print("\n--- HistGradientBoosting: 8 original features (leave-one-target-out) ---")
    X_8 = np.column_stack([
        np.abs(X[:, :7]),  # absolute deltas (7 cols)
        1.0 - X[:, 7],    # FP dissimilarity
    ])
    hgb8_ndcg, hgb8_per_target, hgb8_time = leave_one_target_out_ndcg(X_8, Y)
    print(f"  HGB (8 feat) NDCG@5: {hgb8_ndcg:.4f}  [{hgb8_time:.1f}s total]")
    best_t = max(hgb8_per_target, key=hgb8_per_target.get)
    worst_t = min(hgb8_per_target, key=hgb8_per_target.get)
    print(f"  Best target:  {best_t} ({hgb8_per_target[best_t]:.4f})")
    print(f"  Worst target: {worst_t} ({hgb8_per_target[worst_t]:.4f})")

    # ── HistGradientBoosting on augmented features ───────────────────────
    print("\n--- HistGradientBoosting: augmented features (|delta| + delta + dissim + interactions) ---")
    abs_deltas = np.abs(X[:, :7])
    signed_deltas = X[:, :7]
    dissim = (1.0 - X[:, 7]).reshape(-1, 1)
    # Interactions: |delta_LogP| × dissim, |delta_HBD| × dissim
    interact1 = (abs_deltas[:, 1] * dissim[:, 0]).reshape(-1, 1)
    interact2 = (abs_deltas[:, 3] * dissim[:, 0]).reshape(-1, 1)
    X_aug = np.column_stack([abs_deltas, signed_deltas, dissim, interact1, interact2])
    print(f"  Augmented feature count: {X_aug.shape[1]}")

    hgb_aug_ndcg, hgb_aug_per_target, hgb_aug_time = leave_one_target_out_ndcg(X_aug, Y)
    print(f"  HGB (aug) NDCG@5: {hgb_aug_ndcg:.4f}  [{hgb_aug_time:.1f}s total]")
    best_t = max(hgb_aug_per_target, key=hgb_aug_per_target.get)
    worst_t = min(hgb_aug_per_target, key=hgb_aug_per_target.get)
    print(f"  Best target:  {best_t} ({hgb_aug_per_target[best_t]:.4f})")
    print(f"  Worst target: {worst_t} ({hgb_aug_per_target[worst_t]:.4f})")

    # ── HistGradientBoosting with all 12 features ────────────────────────
    print("\n--- HistGradientBoosting: all 12 features (with env context) ---")
    X_12 = np.column_stack([abs_deltas, dissim, X[:, 8:12]])
    print(f"  Feature count: {X_12.shape[1]}")

    hgb12_ndcg, hgb12_per_target, hgb12_time = leave_one_target_out_ndcg(X_12, Y)
    print(f"  HGB (12 feat) NDCG@5: {hgb12_ndcg:.4f}  [{hgb12_time:.1f}s total]")
    best_t = max(hgb12_per_target, key=hgb12_per_target.get)
    worst_t = min(hgb12_per_target, key=hgb12_per_target.get)
    print(f"  Best target:  {best_t} ({hgb12_per_target[best_t]:.4f})")
    print(f"  Worst target: {worst_t} ({hgb12_per_target[worst_t]:.4f})")

    # ── HistGradientBoosting with deeper trees ───────────────────────────
    print("\n--- HistGradientBoosting: 8 features, deeper (max_depth=10, 500 iter) ---")
    deep_kwargs = {
        "max_iter": 500,
        "max_depth": 10,
        "learning_rate": 0.05,
        "min_samples_leaf": 20,
        "random_state": 42,
    }
    hgb_deep_ndcg, hgb_deep_per_target, hgb_deep_time = leave_one_target_out_ndcg(
        X_8, Y, model_kwargs=deep_kwargs
    )
    print(f"  HGB (deep) NDCG@5: {hgb_deep_ndcg:.4f}  [{hgb_deep_time:.1f}s total]")
    best_t = max(hgb_deep_per_target, key=hgb_deep_per_target.get)
    worst_t = min(hgb_deep_per_target, key=hgb_deep_per_target.get)
    print(f"  Best target:  {best_t} ({hgb_deep_per_target[best_t]:.4f})")
    print(f"  Worst target: {worst_t} ({hgb_deep_per_target[worst_t]:.4f})")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Random baseline:             {random_ndcg:.4f}")
    print(f"  |delta_HAC| only:            {hac_ndcg:.4f}")
    print(f"  L2 norm (best manual):       {l2_ndcg:.4f}")
    print(f"  HGB 8 feat (LOO-target):     {hgb8_ndcg:.4f}")
    print(f"  HGB augmented (LOO-target):  {hgb_aug_ndcg:.4f}")
    print(f"  HGB 12 feat (LOO-target):    {hgb12_ndcg:.4f}")
    print(f"  HGB deep (LOO-target):       {hgb_deep_ndcg:.4f}")
    print()
    print("Interpretation:")
    print("  If HGB ~ L2 norm: feature set is fundamentally limited -> need richer features")
    print("  If HGB >> L2 norm: nonlinear patterns exist -> evolved functions can improve")
    print("  If HGB + env > HGB 8: env context helps with a learned model")


if __name__ == "__main__":
    main()
