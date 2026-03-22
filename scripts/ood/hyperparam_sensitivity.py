#!/usr/bin/env python
"""
OOD Experiment 11: Hyperparameter Sensitivity

Random search over 50 HGB configurations to demonstrate that the
reported NDCG@3 is robust to hyperparameter choices.

Protocol:
  - Random search: max_iter, max_depth, learning_rate, min_samples_leaf
  - LOO-target NDCG@3 for each configuration
  - Report distribution: best, worst, median, std
  - Compare every config against the -core_n_heavy heuristic

Usage:
    python scripts/ood/hyperparam_sensitivity.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

sys.stdout.reconfigure(line_buffering=True)

# ── Load data ────────────────────────────────────────────────────────────────

EVAL_DATA_PATH = Path("evolve/eval_data/position_data.npz")
_data = np.load(EVAL_DATA_PATH, allow_pickle=True)
X = _data["X"]
Y = _data["y"]
GROUPS = _data["groups"]
OFFSETS = _data["target_offsets"]
TARGETS = _data["target_names"]
FEAT_NAMES = list(_data["feature_names"])
del _data

print(f"Loaded: {X.shape[0]:,} rows, {X.shape[1]} features, {len(TARGETS)} targets")

IDX_N_HEAVY = FEAT_NAMES.index("core_n_heavy")


# ── Evaluation functions ─────────────────────────────────────────────────────

def ndcg_at_k(y_true, y_score, k=3):
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


def hit_rate_at_1(y_true, y_score):
    if len(y_true) < 2:
        return 0.0
    return float(np.argmax(y_score) == np.argmax(y_true))


def eval_metrics_for_target(scores, y, groups, k=3):
    ndcgs, hits = [], []
    for mol_id in np.unique(groups):
        mask = groups == mol_id
        if mask.sum() < k:
            continue
        ndcgs.append(ndcg_at_k(y[mask], scores[mask], k))
        hits.append(hit_rate_at_1(y[mask], scores[mask]))
    if len(y) > 5:
        rho, _ = stats.spearmanr(scores, y)
    else:
        rho = 0.0
    return {
        "ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "hit_rate": float(np.mean(hits)) if hits else 0.0,
        "spearman": float(rho) if np.isfinite(rho) else 0.0,
    }


# ── Hyperparameter search ───────────────────────────────────────────────────

PARAM_GRID = {
    "max_iter": [100, 200, 300, 500, 1000],
    "max_depth": [3, 4, 5, 6, 7, 8, 10],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "min_samples_leaf": [10, 20, 30, 50, 100, 200],
}


def loo_target_hgb(hgb_kwargs, label):
    """Run LOO-target evaluation with given HGB hyperparameters."""
    all_ndcgs, all_hits, all_rhos = [], [], []
    t0 = time.perf_counter()

    for i in range(len(TARGETS)):
        lo, hi = OFFSETS[i], OFFSETS[i + 1]
        train_mask = np.ones(len(Y), dtype=bool)
        train_mask[lo:hi] = False

        model = HistGradientBoostingRegressor(**hgb_kwargs)
        model.fit(X[train_mask], Y[train_mask])
        preds = model.predict(X[lo:hi])

        m = eval_metrics_for_target(preds, Y[lo:hi], GROUPS[lo:hi])
        all_ndcgs.append(m["ndcg"])
        all_hits.append(m["hit_rate"])
        all_rhos.append(m["spearman"])

    elapsed = time.perf_counter() - t0
    result = {
        "label": label,
        "params": {k: v for k, v in hgb_kwargs.items() if k != "random_state"},
        "ndcg": float(np.mean(all_ndcgs)),
        "hit_rate": float(np.mean(all_hits)),
        "spearman": float(np.mean(all_rhos)),
        "ndcg_std": float(np.std(all_ndcgs)),
        "elapsed_s": round(elapsed, 1),
    }
    print(f"  {label:35s}  NDCG@3={result['ndcg']:.4f} (+-{result['ndcg_std']:.4f})  "
          f"Hit@1={result['hit_rate']:.3f}  Spearman={result['spearman']:.4f}  "
          f"[{elapsed:.0f}s]")
    return result


def main():
    print("\n" + "=" * 78)
    print("OOD EXPERIMENT 11: HYPERPARAMETER SENSITIVITY")
    print("=" * 78)

    # Heuristic baseline for reference
    print("\n--- Heuristic baseline ---")
    heur_ndcgs = []
    for i in range(len(TARGETS)):
        lo, hi = OFFSETS[i], OFFSETS[i + 1]
        scores = -X[lo:hi, IDX_N_HEAVY]
        m = eval_metrics_for_target(scores, Y[lo:hi], GROUPS[lo:hi])
        heur_ndcgs.append(m["ndcg"])
    heur_ndcg = float(np.mean(heur_ndcgs))
    print(f"  -core_n_heavy heuristic:  NDCG@3={heur_ndcg:.4f}")

    # Default configuration
    print("\n--- Default HGB configuration ---")
    default_kwargs = {
        "max_iter": 300, "max_depth": 6,
        "learning_rate": 0.1, "min_samples_leaf": 50,
        "random_state": 42,
    }
    default_result = loo_target_hgb(default_kwargs, "Default (300/6/0.1/50)")

    # Random search
    n_configs = 50
    print(f"\n--- Random search ({n_configs} configurations) ---")
    rng = np.random.RandomState(42)

    all_results = [default_result]
    for ci in range(n_configs):
        config = {
            "max_iter": int(rng.choice(PARAM_GRID["max_iter"])),
            "max_depth": int(rng.choice(PARAM_GRID["max_depth"])),
            "learning_rate": float(rng.choice(PARAM_GRID["learning_rate"])),
            "min_samples_leaf": int(rng.choice(PARAM_GRID["min_samples_leaf"])),
            "random_state": 42,
        }
        label = (f"Config {ci+1:02d} "
                 f"({config['max_iter']}/{config['max_depth']}/"
                 f"{config['learning_rate']}/{config['min_samples_leaf']})")
        result = loo_target_hgb(config, label)
        all_results.append(result)

    # ── Summary ──────────────────────────────────────────────────────────
    ndcgs = np.array([r["ndcg"] for r in all_results])
    spearman_vals = np.array([r["spearman"] for r in all_results])

    print("\n" + "=" * 78)
    print("HYPERPARAMETER SENSITIVITY SUMMARY")
    print("=" * 78)

    print(f"\n  Configurations tested: {len(all_results)} "
          f"(1 default + {n_configs} random)")

    print(f"\n  NDCG@3 distribution:")
    print(f"    mean   = {np.mean(ndcgs):.4f}")
    print(f"    std    = {np.std(ndcgs):.4f}")
    print(f"    min    = {np.min(ndcgs):.4f}")
    print(f"    p25    = {np.percentile(ndcgs, 25):.4f}")
    print(f"    median = {np.median(ndcgs):.4f}")
    print(f"    p75    = {np.percentile(ndcgs, 75):.4f}")
    print(f"    max    = {np.max(ndcgs):.4f}")
    print(f"    range  = {np.max(ndcgs) - np.min(ndcgs):.4f}")

    print(f"\n  Spearman distribution:")
    print(f"    mean   = {np.mean(spearman_vals):.4f}")
    print(f"    std    = {np.std(spearman_vals):.4f}")
    print(f"    range  = {np.max(spearman_vals) - np.min(spearman_vals):.4f}")

    n_beat_heur = int(np.sum(ndcgs > heur_ndcg))
    print(f"\n  -core_n_heavy heuristic NDCG@3: {heur_ndcg:.4f}")
    print(f"  Configs beating heuristic: {n_beat_heur}/{len(all_results)}")

    # Top/bottom configs
    sorted_results = sorted(all_results, key=lambda r: r["ndcg"], reverse=True)
    print(f"\n  Top 5 configurations:")
    for r in sorted_results[:5]:
        print(f"    NDCG@3={r['ndcg']:.4f}  Spearman={r['spearman']:.4f}  {r['params']}")
    print(f"\n  Bottom 5 configurations:")
    for r in sorted_results[-5:]:
        print(f"    NDCG@3={r['ndcg']:.4f}  Spearman={r['spearman']:.4f}  {r['params']}")

    # Interpretation
    ndcg_range = float(np.max(ndcgs) - np.min(ndcgs))
    ndcg_std = float(np.std(ndcgs))
    print(f"\n  INTERPRETATION:")
    if ndcg_range < 0.01:
        print(f"  -> VERY ROBUST: NDCG@3 range={ndcg_range:.4f} (< 0.01)")
    elif ndcg_range < 0.02:
        print(f"  -> ROBUST: NDCG@3 range={ndcg_range:.4f} (< 0.02)")
    elif ndcg_range < 0.05:
        print(f"  -> MODERATELY ROBUST: NDCG@3 range={ndcg_range:.4f}")
    else:
        print(f"  -> SENSITIVE: NDCG@3 range={ndcg_range:.4f} (>= 0.05)")
        print(f"    Some configs underperform substantially")

    if n_beat_heur == 0:
        print(f"  -> NO config beats the heuristic — confirms core_n_heavy dominance")

    # ── Save ─────────────────────────────────────────────────────────────
    out_path = Path("outputs/ood/hyperparam_sensitivity_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "heuristic_ndcg": heur_ndcg,
        "results": all_results,
        "summary": {
            "n_configs": len(all_results),
            "ndcg_mean": float(np.mean(ndcgs)),
            "ndcg_std": ndcg_std,
            "ndcg_min": float(np.min(ndcgs)),
            "ndcg_p25": float(np.percentile(ndcgs, 25)),
            "ndcg_median": float(np.median(ndcgs)),
            "ndcg_p75": float(np.percentile(ndcgs, 75)),
            "ndcg_max": float(np.max(ndcgs)),
            "ndcg_range": ndcg_range,
            "spearman_mean": float(np.mean(spearman_vals)),
            "spearman_std": float(np.std(spearman_vals)),
            "n_beat_heuristic": n_beat_heur,
        },
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
