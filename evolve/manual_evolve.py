"""
Manual evolution harness: evaluate candidate scoring functions without
needing an external LLM API. Claude Code acts as the mutation operator.

Usage:
    python evolve/manual_evolve.py evolve/candidates/gen1_v1.py
    python evolve/manual_evolve.py evolve/candidates/  # evaluate all .py files in dir
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

# Load eval data
EVAL_DATA_PATH = Path(__file__).parent / "eval_data" / "eval_data.npz"
_data = np.load(EVAL_DATA_PATH, allow_pickle=True)
X = _data["X"]
Y = _data["y"]
GROUPS = _data["groups"]
OFFSETS = _data["target_offsets"]
TARGETS = _data["target_names"]
del _data


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


def evaluate_function(score_fn):
    """Run the scoring function and compute NDCG@5 across all targets."""
    t0 = time.perf_counter()
    try:
        scores = score_fn(X)
    except Exception as e:
        return {"error": str(e)}

    if not isinstance(scores, np.ndarray) or scores.shape != (X.shape[0],):
        return {"error": f"Bad shape: {getattr(scores, 'shape', type(scores))}"}
    if np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
        return {"error": "NaN or Inf in scores"}

    eval_time = time.perf_counter() - t0

    target_ndcgs = {}
    for i, target in enumerate(TARGETS):
        lo, hi = OFFSETS[i], OFFSETS[i + 1]
        s, y, g = scores[lo:hi], Y[lo:hi], GROUPS[lo:hi]
        ndcgs = []
        for mol_id in np.unique(g):
            mask = g == mol_id
            if mask.sum() >= 5:
                ndcgs.append(ndcg_at_k(y[mask], s[mask], 5))
        if ndcgs:
            target_ndcgs[str(target)] = float(np.mean(ndcgs))

    mean = float(np.mean(list(target_ndcgs.values()))) if target_ndcgs else 0.0
    best_t = max(target_ndcgs, key=target_ndcgs.get) if target_ndcgs else ""
    worst_t = min(target_ndcgs, key=target_ndcgs.get) if target_ndcgs else ""

    return {
        "mean_ndcg5": round(mean, 4),
        "best_target": f"{best_t} ({target_ndcgs.get(best_t, 0):.4f})",
        "worst_target": f"{worst_t} ({target_ndcgs.get(worst_t, 0):.4f})",
        "eval_time": round(eval_time, 3),
        "num_targets": len(target_ndcgs),
    }


def load_and_eval(path: Path):
    """Import score_transformations from a .py file and evaluate it."""
    spec = importlib.util.spec_from_file_location("candidate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "score_transformations", None)
    if fn is None:
        return {"error": "No score_transformations function found"}
    return evaluate_function(fn)


def main():
    paths = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.py")))
        elif p.is_file():
            paths.append(p)

    if not paths:
        print("Usage: python evolve/manual_evolve.py <file_or_dir> [...]")
        sys.exit(1)

    results = []
    for p in paths:
        r = load_and_eval(p)
        r["file"] = p.name
        results.append(r)
        ndcg = r.get("mean_ndcg5", "ERROR")
        err = r.get("error", "")
        tag = f"  ERROR: {err}" if err else ""
        print(f"  {p.name:40s}  NDCG@5 = {ndcg}{tag}")

    # Summary
    valid = [r for r in results if "error" not in r]
    if valid:
        best = max(valid, key=lambda r: r["mean_ndcg5"])
        print(f"\n  BEST: {best['file']}  NDCG@5 = {best['mean_ndcg5']}")


if __name__ == "__main__":
    main()
