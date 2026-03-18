"""
ShinkaEvolve fitness evaluator for transformation scoring functions.

Computes leave-one-target-out NDCG@5: for each target, for each starting
molecule, score all known transformations and measure how well the top-5
predicted modifications match the actual largest activity changes.

This file follows the ShinkaEvolve evaluate.py convention:
  - main(program_path, results_dir) is called by the framework
  - run_shinka_eval handles importing the evolved function and orchestration
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from shinka.core import run_shinka_eval

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

EVAL_DATA_PATH = Path(__file__).parent / "eval_data" / "eval_data.npz"
NDCG_K = 5

# ── Load evaluation data once at module level ─────────────────────────────────

_data = np.load(EVAL_DATA_PATH, allow_pickle=True)
EVAL_X = _data["X"]                          # (N, 8) float32
EVAL_Y = _data["y"]                          # (N,) float32 — abs_delta_pActivity
EVAL_GROUPS = _data["groups"]                # (N,) int64 — mol_from IDs
EVAL_TARGET_OFFSETS = _data["target_offsets"] # (T+1,) int64
EVAL_TARGET_NAMES = _data["target_names"]    # (T,) str
del _data

logger.info(
    "Eval data loaded: %d rows, %d targets", EVAL_X.shape[0], len(EVAL_TARGET_NAMES)
)


# ── NDCG@k implementation ────────────────────────────────────────────────────

def _ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """
    Compute NDCG@k for a single query (starting molecule).

    y_true  : graded relevance (abs_delta_pActivity)
    y_score : predicted scores from the evolved function
    k       : cutoff
    """
    n = len(y_true)
    if n == 0:
        return 0.0
    k = min(k, n)

    # DCG: rank by predicted score
    pred_order = np.argsort(-y_score)[:k]
    discounts = np.log2(np.arange(2, k + 2))
    dcg = np.sum(y_true[pred_order] / discounts)

    # IDCG: rank by true relevance (best possible)
    ideal_order = np.argsort(-y_true)[:k]
    idcg = np.sum(y_true[ideal_order] / discounts)

    if idcg <= 0:
        return 0.0
    return float(dcg / idcg)


def _evaluate_target(
    scores: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    k: int,
) -> tuple[float, int]:
    """
    Compute mean NDCG@k across all starting molecules for one target.

    Returns (mean_ndcg, num_molecules_evaluated).
    """
    ndcgs: list[float] = []
    for mol_id in np.unique(groups):
        mask = groups == mol_id
        if mask.sum() < k:
            continue
        ndcgs.append(_ndcg_at_k(y[mask], scores[mask], k))

    if not ndcgs:
        return 0.0, 0
    return float(np.mean(ndcgs)), len(ndcgs)


# ── ShinkaEvolve interface functions ──────────────────────────────────────────

def get_kwargs(run_idx: int) -> dict:
    """Provide the feature matrix to the evolved scoring function."""
    return {"X": EVAL_X}


def validate(result) -> tuple[bool, str | None]:
    """Check that the scoring function returned a valid array."""
    if not isinstance(result, np.ndarray):
        return False, f"Expected np.ndarray, got {type(result).__name__}"

    expected_shape = (EVAL_X.shape[0],)
    if result.shape != expected_shape:
        return False, f"Expected shape {expected_shape}, got {result.shape}"

    if np.any(np.isnan(result)):
        return False, "Scores contain NaN values"

    if np.any(np.isinf(result)):
        return False, "Scores contain Inf values"

    return True, None


def aggregate(results: list) -> dict:
    """
    Compute leave-one-target-out NDCG@5 from the scored transformations.

    Since the evolved function uses only molecular features (no target identity),
    evaluation on each target IS leave-one-target-out: the function never saw
    target-specific information during its construction.
    """
    scores_all = results[0]  # single run

    target_ndcgs: dict[str, float] = {}
    target_counts: dict[str, int] = {}

    for i, target in enumerate(EVAL_TARGET_NAMES):
        lo = EVAL_TARGET_OFFSETS[i]
        hi = EVAL_TARGET_OFFSETS[i + 1]

        scores_t = scores_all[lo:hi]
        y_t = EVAL_Y[lo:hi]
        groups_t = EVAL_GROUPS[lo:hi]

        mean_ndcg, n_mols = _evaluate_target(scores_t, y_t, groups_t, NDCG_K)
        if n_mols > 0:
            target_ndcgs[target] = mean_ndcg
            target_counts[target] = n_mols

    if not target_ndcgs:
        mean_ndcg = 0.0
    else:
        mean_ndcg = float(np.mean(list(target_ndcgs.values())))

    # Also compute cliff-recall@5: fraction of top-5 that are actual cliffs
    cliff_threshold = 1.5
    cliff_recall_targets: list[float] = []
    for i, target in enumerate(EVAL_TARGET_NAMES):
        lo = EVAL_TARGET_OFFSETS[i]
        hi = EVAL_TARGET_OFFSETS[i + 1]
        scores_t = scores_all[lo:hi]
        y_t = EVAL_Y[lo:hi]
        groups_t = EVAL_GROUPS[lo:hi]

        recalls: list[float] = []
        for mol_id in np.unique(groups_t):
            mask = groups_t == mol_id
            if mask.sum() < NDCG_K:
                continue
            order = np.argsort(-scores_t[mask])[:NDCG_K]
            cliffs_in_top = (y_t[mask][order] >= cliff_threshold).sum()
            total_cliffs = (y_t[mask] >= cliff_threshold).sum()
            if total_cliffs > 0:
                recalls.append(cliffs_in_top / min(NDCG_K, total_cliffs))
        if recalls:
            cliff_recall_targets.append(float(np.mean(recalls)))

    mean_cliff_recall = float(np.mean(cliff_recall_targets)) if cliff_recall_targets else 0.0

    return {
        "combined_score": mean_ndcg,
        "public": {
            "mean_ndcg5": round(mean_ndcg, 4),
            "mean_cliff_recall5": round(mean_cliff_recall, 4),
            "num_targets": len(target_ndcgs),
        },
        "private": {
            "per_target_ndcg5": {k: round(v, 4) for k, v in target_ndcgs.items()},
        },
    }


# ── Entry point for ShinkaEvolve ─────────────────────────────────────────────

def main(program_path: str, results_dir: str):
    """Called by ShinkaEvolve framework."""
    return run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="score_transformations",
        num_runs=1,
        run_workers=1,
        get_experiment_kwargs=get_kwargs,
        validate_fn=validate,
        aggregate_metrics_fn=aggregate,
    )
