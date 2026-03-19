"""
M7a: Position-level SAR sensitivity ceiling test.

Tests whether 3D pharmacophore context at the attachment point predicts
which positions on a molecule are most SAR-sensitive.

Hypothesis: by reframing from transformation-level (rank R-group swaps)
to position-level (rank attachment points), 3D context features become
the primary signal rather than dead weight.

Feature sets tested:
  A. Random baseline
  B. Mean sensitivity baseline (predict global mean for all positions)
  C. Core topology only (n_heavy, n_rings)          — 2 features
  D. 3D context only (9 features)                   — the main hypothesis
  E. 3D context + core topology (11 features)       — full feature set
  F. Deeper HGB on full features (11)               — capacity test

Evaluation:
  - Leave-one-target-out
  - NDCG@3: rank positions within each (mol_from, target) group
  - Spearman correlation: predicted vs actual sensitivity at held-out target
  - Hit rate @1: fraction where top-predicted position is the most sensitive

Usage:
    python evolve/position_ceiling.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

sys.stdout.reconfigure(line_buffering=True)

# ── Load position-level eval data ────────────────────────────────────────────

EVAL_DATA_PATH = Path(__file__).parent / "eval_data" / "position_data.npz"
_data = np.load(EVAL_DATA_PATH, allow_pickle=True)
X = _data["X"]
Y = _data["y"]
GROUPS = _data["groups"]
OFFSETS = _data["target_offsets"]
TARGETS = _data["target_names"]
FEAT_NAMES = list(_data["feature_names"])
del _data

print(f"Loaded: {X.shape[0]:,} rows, {X.shape[1]} features, "
      f"{len(TARGETS)} targets")
print(f"Features: {FEAT_NAMES}")

# Feature slices
N_CTX = 9
SL_CTX = slice(0, N_CTX)
SL_TOPO = slice(N_CTX, N_CTX + 2)


# ── Evaluation functions ─────────────────────────────────────────────────────

def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 3) -> float:
    """Normalised discounted cumulative gain at k."""
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


def hit_rate_at_1(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """1.0 if top-predicted position is the most sensitive, else 0.0."""
    if len(y_true) < 2:
        return 0.0
    return float(np.argmax(y_score) == np.argmax(y_true))


def eval_metrics_for_target(
    scores: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    k: int = 3,
) -> dict:
    """Compute NDCG@k, hit rate @1, and Spearman rho within a target."""
    ndcgs = []
    hits = []

    for mol_id in np.unique(groups):
        mask = groups == mol_id
        n = mask.sum()
        if n < k:
            continue
        ndcgs.append(ndcg_at_k(y[mask], scores[mask], k))
        hits.append(hit_rate_at_1(y[mask], scores[mask]))

    # Spearman across ALL positions at this target
    if len(y) > 5:
        rho, _ = stats.spearmanr(scores, y)
    else:
        rho = 0.0

    return {
        "ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "hit_rate": float(np.mean(hits)) if hits else 0.0,
        "spearman": float(rho) if np.isfinite(rho) else 0.0,
        "n_groups": len(ndcgs),
    }


# ── Scoring helpers ──────────────────────────────────────────────────────────

def score_with_function(score_fn, label: str = "") -> dict:
    """Evaluate a non-ML scoring function across all targets."""
    scores = score_fn(X)
    all_ndcgs = []
    all_hits = []
    all_rhos = []

    for i in range(len(TARGETS)):
        lo, hi = OFFSETS[i], OFFSETS[i + 1]
        m = eval_metrics_for_target(
            scores[lo:hi], Y[lo:hi], GROUPS[lo:hi], k=3,
        )
        all_ndcgs.append(m["ndcg"])
        all_hits.append(m["hit_rate"])
        all_rhos.append(m["spearman"])

    result = {
        "ndcg": float(np.mean(all_ndcgs)),
        "hit_rate": float(np.mean(all_hits)),
        "spearman": float(np.mean(all_rhos)),
    }
    print(f"  {label:50s}  NDCG@3={result['ndcg']:.4f}  "
          f"Hit@1={result['hit_rate']:.3f}  "
          f"Spearman={result['spearman']:.4f}")
    return result


def leave_one_target_out(
    X_in: np.ndarray,
    y_in: np.ndarray,
    model_kwargs: dict | None = None,
    label: str = "",
) -> dict:
    """Leave-one-target-out HGB evaluation."""
    if model_kwargs is None:
        model_kwargs = {
            "max_iter": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
            "min_samples_leaf": 50,
            "random_state": 42,
        }

    all_ndcgs = []
    all_hits = []
    all_rhos = []
    t0 = time.perf_counter()

    for i in range(len(TARGETS)):
        lo, hi = OFFSETS[i], OFFSETS[i + 1]
        X_test, y_test = X_in[lo:hi], y_in[lo:hi]
        g_test = GROUPS[lo:hi]

        train_mask = np.ones(len(y_in), dtype=bool)
        train_mask[lo:hi] = False
        X_train, y_train = X_in[train_mask], y_in[train_mask]

        model = HistGradientBoostingRegressor(**model_kwargs)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        m = eval_metrics_for_target(preds, y_test, g_test, k=3)
        all_ndcgs.append(m["ndcg"])
        all_hits.append(m["hit_rate"])
        all_rhos.append(m["spearman"])

    elapsed = time.perf_counter() - t0
    result = {
        "ndcg": float(np.mean(all_ndcgs)),
        "hit_rate": float(np.mean(all_hits)),
        "spearman": float(np.mean(all_rhos)),
        "ndcg_std": float(np.std(all_ndcgs)),
        "ndcgs_per_target": dict(zip([str(t) for t in TARGETS], all_ndcgs)),
    }

    # Find best/worst targets
    best_i = int(np.argmax(all_ndcgs))
    worst_i = int(np.argmin(all_ndcgs))

    print(f"  {label:50s}  NDCG@3={result['ndcg']:.4f}  "
          f"Hit@1={result['hit_rate']:.3f}  "
          f"Spearman={result['spearman']:.4f}  [{elapsed:.0f}s]")
    print(f"    best: {TARGETS[best_i]} ({all_ndcgs[best_i]:.4f})  "
          f"worst: {TARGETS[worst_i]} ({all_ndcgs[worst_i]:.4f})  "
          f"std: {result['ndcg_std']:.4f}")
    return result


def main():
    print("\n" + "=" * 78)
    print("M7a: Position-Level SAR Sensitivity — Ceiling Test")
    print("=" * 78)
    print(f"\nQuestion: Can 3D context features predict which positions")
    print(f"on a molecule are most SAR-sensitive?")
    print(f"\nOld ceiling (transformation-level): NDCG@5 = 0.5170")
    print(f"Random baseline (transformation):   NDCG@5 = 0.439")

    # ── Baselines (no ML) ────────────────────────────────────────────────
    print("\n--- Baselines (no training) ---")

    # A. Random
    score_with_function(
        lambda _: np.random.RandomState(42).randn(len(Y)),
        "A. Random",
    )

    # B. Global mean (predicts same value for everything)
    score_with_function(
        lambda _: np.full(len(Y), Y.mean()),
        "B. Global mean (constant prediction)",
    )

    # C. Core size heuristic (larger cores → more sensitive?)
    score_with_function(
        lambda x: x[:, -2],  # core_n_heavy
        "C. Core size heuristic (n_heavy)",
    )

    # D. Steric crowding heuristic
    score_with_function(
        lambda x: x[:, 8],  # n_heavy_4A
        "D. Steric crowding heuristic (n_heavy_4A)",
    )

    # E. SASA heuristic (exposed positions → less sensitive?)
    score_with_function(
        lambda x: -x[:, 4],  # sasa_attach (negative → buried = sensitive)
        "E. Buried position heuristic (-SASA)",
    )

    # ── HGB with different feature sets ──────────────────────────────────
    print("\n--- HistGradientBoosting (leave-one-target-out, NDCG@3) ---")

    # F. Core topology only
    X_topo = X[:, SL_TOPO]
    leave_one_target_out(X_topo, Y, label="F. Core topology only (2)")

    # G. 3D context only — THE MAIN HYPOTHESIS
    X_ctx = X[:, SL_CTX]
    leave_one_target_out(X_ctx, Y, label="G. 3D context only (9) *** KEY ***")

    # H. 3D context + topology
    leave_one_target_out(X, Y, label="H. 3D context + topology (11)")

    # ── Deeper HGB ───────────────────────────────────────────────────────
    print("\n--- Deeper HGB ---")
    deep_kwargs = {
        "max_iter": 500,
        "max_depth": 8,
        "learning_rate": 0.05,
        "min_samples_leaf": 30,
        "random_state": 42,
    }
    leave_one_target_out(
        X, Y, model_kwargs=deep_kwargs,
        label="I. Deeper HGB, all features (11)",
    )

    # Very deep
    vdeep_kwargs = {
        "max_iter": 1000,
        "max_depth": 10,
        "learning_rate": 0.03,
        "min_samples_leaf": 20,
        "random_state": 42,
    }
    leave_one_target_out(
        X, Y, model_kwargs=vdeep_kwargs,
        label="J. Very deep HGB, all features (11)",
    )

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("INTERPRETATION GUIDE:")
    print("  - If G (3D context) >> A (random): position-level reframe works!")
    print("    3D features predict position sensitivity across targets.")
    print("  - If G ~ A: position sensitivity is target-specific, need")
    print("    pharmacophore homology grouping (Phase 3).")
    print("  - Compare NDCG@3 values here against transformation-level 0.52.")
    print("    Note: different metrics (NDCG@3 vs @5, different groupings)")
    print("    so absolute values aren't directly comparable, but relative")
    print("    improvement over random is the key signal.")
    print("=" * 78)


if __name__ == "__main__":
    main()
