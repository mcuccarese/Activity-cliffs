#!/usr/bin/env python
"""
M7b: Pharmacophore Homology Grouping.

Tests whether grouping targets by SAR-profile similarity improves
position-level sensitivity predictions beyond the target-agnostic
baseline (M7a: NDCG@3=0.964, Hit@1=57%, Spearman=0.607).

Steps:
  1. Build SAR profile vectors per target from MMP data + R-group properties.
     For each change-type category (EWG gain, EDG gain, size increase, etc.),
     compute the mean cliff rate and mean |delta_p| at that target.
  2. Compute pairwise target-target Pearson correlation of profile vectors.
     Visualize as a heatmap.
  3. Cluster targets using hierarchical clustering (Ward linkage).
  4. Test cluster-conditioned position sensitivity:
     a) Global model + cluster_id as feature
     b) Separate models per cluster (within-cluster leave-one-target-out)

Usage:
    python scripts/pharmacophore_homology.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.ensemble import HistGradientBoostingRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)
sys.stdout.reconfigure(line_buffering=True)

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MMPS_PATH = PROJECT_ROOT / "outputs" / "mmps" / "all_mmps.parquet"
RGROUP_PROPS_PATH = PROJECT_ROOT / "outputs" / "features" / "rgroup_props.parquet"
POSITION_DATA_PATH = PROJECT_ROOT / "evolve" / "eval_data" / "position_data.npz"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pharmacophore_homology"

# Change-type categories for SAR profiling
# Each category is defined by a condition on the delta properties
# delta = props(rgroup_to) - props(rgroup_from)
CHANGE_CATEGORIES = {
    "ewg_gain": ("delta_has_ewg", ">", 0),
    "ewg_loss": ("delta_has_ewg", "<", 0),
    "edg_gain": ("delta_has_edg", ">", 0),
    "edg_loss": ("delta_has_edg", "<", 0),
    "size_increase": ("delta_heavy_atoms", ">", 1),
    "size_decrease": ("delta_heavy_atoms", "<", -1),
    "ring_gain": ("delta_n_rings", ">", 0),
    "ring_loss": ("delta_n_rings", "<", 0),
    "lipophilicity_increase": ("delta_lipophilicity", ">", 0.5),
    "lipophilicity_decrease": ("delta_lipophilicity", "<", -0.5),
    "hbd_gain": ("delta_n_hbd", ">", 0),
    "hba_gain": ("delta_n_hba", ">", 0),
    "aromaticity_gain": ("delta_n_arom_rings", ">", 0),
    "aromaticity_loss": ("delta_n_arom_rings", "<", 0),
}

# R-group property names (must match rgroup_props.parquet columns)
PROP_NAMES = [
    "has_ewg", "has_edg", "ewg_count", "edg_count",
    "n_hbd", "n_hba", "lipophilicity", "heavy_atoms",
    "n_rings", "n_arom_rings", "fsp3",
]


# ── Evaluation functions (from position_ceiling.py) ──────────────────────────

def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 3) -> float:
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
    if len(y_true) < 2:
        return 0.0
    return float(np.argmax(y_score) == np.argmax(y_true))


def eval_metrics_for_target(
    scores: np.ndarray, y: np.ndarray, groups: np.ndarray, k: int = 3,
) -> dict:
    ndcgs, hits = [], []
    for mol_id in np.unique(groups):
        mask = groups == mol_id
        n = mask.sum()
        if n < k:
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
        "n_groups": len(ndcgs),
    }


# ── Step 1: Build SAR profile vectors ───────────────────────────────────────

def build_sar_profiles(
    mmps: pd.DataFrame, rg_props: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build a SAR profile vector per target.

    For each change-type category, compute:
      - cliff_rate: fraction of MMPs with |delta_p| > 1.5
      - mean_abs_delta: mean |delta_p|

    Returns profile DataFrame (targets x profile dims) and target list.
    """
    logger.info("Building R-group property lookup ...")
    prop_lookup = dict(zip(
        rg_props["rgroup_smiles"],
        rg_props[PROP_NAMES].values.astype(np.float32),
    ))
    logger.info("  %d R-groups in lookup", len(prop_lookup))

    # Compute delta properties for each MMP
    logger.info("Computing delta properties for %s MMPs ...", f"{len(mmps):,}")
    zero_props = np.zeros(len(PROP_NAMES), dtype=np.float32)

    props_from = np.array([
        prop_lookup.get(s, zero_props) for s in mmps["rgroup_from"]
    ], dtype=np.float32)
    props_to = np.array([
        prop_lookup.get(s, zero_props) for s in mmps["rgroup_to"]
    ], dtype=np.float32)
    deltas = props_to - props_from

    # Build a dataframe with delta properties
    delta_cols = [f"delta_{n}" for n in PROP_NAMES]
    delta_df = pd.DataFrame(deltas, columns=delta_cols)
    delta_df["target_chembl_id"] = mmps["target_chembl_id"].values
    delta_df["abs_delta_p"] = mmps["abs_delta_pActivity"].values
    delta_df["is_cliff"] = (delta_df["abs_delta_p"] > 1.5).astype(np.float32)

    # Build profile per target
    logger.info("Building SAR profiles per target ...")
    targets = sorted(delta_df["target_chembl_id"].unique())
    profile_cols = []
    for cat_name in CHANGE_CATEGORIES:
        profile_cols.append(f"{cat_name}_cliff_rate")
        profile_cols.append(f"{cat_name}_mean_abs_delta")

    profiles = np.zeros((len(targets), len(profile_cols)), dtype=np.float64)

    for t_idx, target in enumerate(targets):
        t_mask = delta_df["target_chembl_id"] == target
        t_data = delta_df[t_mask]

        col_idx = 0
        for cat_name, (col, op, threshold) in CHANGE_CATEGORIES.items():
            if op == ">":
                cat_mask = t_data[col] > threshold
            else:
                cat_mask = t_data[col] < threshold

            cat_data = t_data[cat_mask]
            if len(cat_data) > 0:
                profiles[t_idx, col_idx] = cat_data["is_cliff"].mean()
                profiles[t_idx, col_idx + 1] = cat_data["abs_delta_p"].mean()
            col_idx += 2

    profile_df = pd.DataFrame(profiles, columns=profile_cols, index=targets)
    logger.info(
        "  SAR profiles: %d targets x %d dimensions",
        len(targets), len(profile_cols),
    )

    return profile_df, targets


# ── Step 2: Correlation heatmap ──────────────────────────────────────────────

def compute_correlation_heatmap(
    profile_df: pd.DataFrame, output_dir: Path,
) -> np.ndarray:
    """Compute pairwise Pearson correlations and save heatmap."""
    logger.info("Computing pairwise Pearson correlations ...")
    corr = profile_df.T.corr(method="pearson")
    corr_vals = corr.values

    # Save correlation matrix
    corr.to_csv(output_dir / "target_correlations.csv")

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr_vals, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    # Shorten target labels for readability
    short_labels = [t.replace("CHEMBL", "C") for t in corr.index]
    ax.set_xticklabels(short_labels, rotation=90, fontsize=6)
    ax.set_yticklabels(short_labels, fontsize=6)
    ax.set_title("Target-Target SAR Profile Correlation (Pearson)")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    plt.tight_layout()
    fig.savefig(output_dir / "sar_profile_correlation_heatmap.png", dpi=150)
    plt.close(fig)
    logger.info("  Saved heatmap to %s", output_dir / "sar_profile_correlation_heatmap.png")

    # Print summary statistics
    upper_tri = corr_vals[np.triu_indices_from(corr_vals, k=1)]
    print(f"\n  Correlation summary (upper triangle, {len(upper_tri)} pairs):")
    print(f"    mean={np.mean(upper_tri):.3f}  std={np.std(upper_tri):.3f}")
    print(f"    min={np.min(upper_tri):.3f}  max={np.max(upper_tri):.3f}")
    print(f"    median={np.median(upper_tri):.3f}")

    return corr_vals


# ── Step 3: Hierarchical clustering ──────────────────────────────────────────

def cluster_targets(
    corr_vals: np.ndarray,
    targets: list[str],
    output_dir: Path,
    n_clusters_range: tuple[int, int] = (5, 8),
) -> dict[int, np.ndarray]:
    """
    Cluster targets using hierarchical clustering (Ward linkage).

    Returns dict mapping n_clusters -> cluster assignments array.
    """
    logger.info("Clustering targets (Ward linkage) ...")

    # Convert correlation to distance: d = 1 - r
    dist_mat = 1.0 - corr_vals
    np.fill_diagonal(dist_mat, 0.0)
    # Ensure symmetry and non-negative
    dist_mat = (dist_mat + dist_mat.T) / 2
    dist_mat = np.clip(dist_mat, 0, 2)

    condensed = squareform(dist_mat)
    Z = linkage(condensed, method="ward")

    # Save dendrogram
    fig, ax = plt.subplots(figsize=(16, 8))
    short_labels = [t.replace("CHEMBL", "C") for t in targets]
    dendrogram(Z, labels=short_labels, leaf_rotation=90, leaf_font_size=7, ax=ax)
    ax.set_title("Target Clustering by SAR Profile (Ward Linkage)")
    ax.set_ylabel("Distance (1 - Pearson r)")
    plt.tight_layout()
    fig.savefig(output_dir / "target_dendrogram.png", dpi=150)
    plt.close(fig)
    logger.info("  Saved dendrogram to %s", output_dir / "target_dendrogram.png")

    # Cut at different k values
    cluster_assignments = {}
    for k in range(n_clusters_range[0], n_clusters_range[1] + 1):
        labels = fcluster(Z, t=k, criterion="maxclust")
        cluster_assignments[k] = labels
        # Print cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        print(f"  k={k}: cluster sizes = {dict(zip(unique.tolist(), counts.tolist()))}")

    # Save cluster assignments
    cluster_df = pd.DataFrame({"target_chembl_id": targets})
    for k, labels in cluster_assignments.items():
        cluster_df[f"cluster_k{k}"] = labels
    cluster_df.to_csv(output_dir / "cluster_assignments.csv", index=False)
    logger.info("  Saved cluster assignments to %s", output_dir / "cluster_assignments.csv")

    return cluster_assignments


# ── Step 4+5: Evaluate cluster-conditioned models ───────────────────────────

def leave_one_target_out(
    X_in: np.ndarray,
    y_in: np.ndarray,
    offsets: np.ndarray,
    groups: np.ndarray,
    targets: np.ndarray,
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

    all_ndcgs, all_hits, all_rhos = [], [], []
    t0 = time.perf_counter()

    for i in range(len(targets)):
        lo, hi = offsets[i], offsets[i + 1]
        X_test, y_test = X_in[lo:hi], y_in[lo:hi]
        g_test = groups[lo:hi]

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
    }

    print(f"  {label:55s}  NDCG@3={result['ndcg']:.4f}  "
          f"Hit@1={result['hit_rate']:.3f}  "
          f"Spearman={result['spearman']:.4f}  [{elapsed:.0f}s]")
    return result


def evaluate_cluster_conditioned(
    position_data_path: Path,
    cluster_assignments: dict[int, np.ndarray],
    targets_from_profiles: list[str],
    output_dir: Path,
) -> dict:
    """
    Test whether cluster information improves position-level predictions.

    Experiment A: Add cluster_id as a categorical feature to the global model.
    Experiment B: Train separate models per cluster (within-cluster LOO).
    """
    # Load position data
    logger.info("Loading position data from %s ...", position_data_path)
    data = np.load(position_data_path, allow_pickle=True)
    X = data["X"]
    Y = data["y"]
    GROUPS = data["groups"]
    OFFSETS = data["target_offsets"]
    TARGETS = list(data["target_names"])
    FEAT_NAMES = list(data["feature_names"])
    del data

    print(f"\nPosition data: {X.shape[0]:,} rows, {X.shape[1]} features, "
          f"{len(TARGETS)} targets")

    # Map target names from position data to cluster labels
    # Need to align targets between profile data and position data
    target_to_idx = {t: i for i, t in enumerate(targets_from_profiles)}

    results = {}

    # ── Baseline: reproduce M7a result ───────────────────────────────────
    print("\n--- Baseline (M7a reproduction) ---")
    baseline = leave_one_target_out(
        X, Y, OFFSETS, GROUPS, TARGETS,
        label="Baseline: 11 features, no cluster info",
    )
    results["baseline"] = baseline

    # ── Experiment A: Global model + cluster_id feature ──────────────────
    print("\n--- Experiment A: Global model + cluster_id as feature ---")

    for k, labels in sorted(cluster_assignments.items()):
        # Build cluster_id vector aligned with position data
        cluster_col = np.zeros(len(Y), dtype=np.float32)
        for i, target in enumerate(TARGETS):
            lo, hi = OFFSETS[i], OFFSETS[i + 1]
            prof_idx = target_to_idx.get(target)
            if prof_idx is not None:
                cluster_col[lo:hi] = labels[prof_idx]
            else:
                cluster_col[lo:hi] = 0  # unknown target

        X_aug = np.column_stack([X, cluster_col])
        res = leave_one_target_out(
            X_aug, Y, OFFSETS, GROUPS, TARGETS,
            label=f"A. Global + cluster_id (k={k}, {X_aug.shape[1]} feat)",
        )
        results[f"global_cluster_k{k}"] = res

    # ── Experiment A2: One-hot encoded cluster IDs ───────────────────────
    print("\n--- Experiment A2: Global model + one-hot cluster IDs ---")

    for k, labels in sorted(cluster_assignments.items()):
        # Build one-hot cluster vectors
        n_clusters = len(np.unique(labels))
        onehot = np.zeros((len(Y), n_clusters), dtype=np.float32)
        for i, target in enumerate(TARGETS):
            lo, hi = OFFSETS[i], OFFSETS[i + 1]
            prof_idx = target_to_idx.get(target)
            if prof_idx is not None:
                cl = labels[prof_idx] - 1  # fcluster is 1-indexed
                onehot[lo:hi, cl] = 1.0

        X_aug = np.column_stack([X, onehot])
        res = leave_one_target_out(
            X_aug, Y, OFFSETS, GROUPS, TARGETS,
            label=f"A2. Global + one-hot cluster (k={k}, {X_aug.shape[1]} feat)",
        )
        results[f"global_onehot_k{k}"] = res

    # ── Experiment B: Separate models per cluster ────────────────────────
    print("\n--- Experiment B: Within-cluster separate models ---")

    model_kwargs = {
        "max_iter": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
        "min_samples_leaf": 50,
        "random_state": 42,
    }

    for k, labels in sorted(cluster_assignments.items()):
        # Map targets to clusters
        target_cluster = {}
        for i, target in enumerate(targets_from_profiles):
            target_cluster[target] = labels[i]

        all_ndcgs, all_hits, all_rhos = [], [], []
        t0 = time.perf_counter()

        for i, target in enumerate(TARGETS):
            lo, hi = OFFSETS[i], OFFSETS[i + 1]
            X_test, y_test = X[lo:hi], Y[lo:hi]
            g_test = GROUPS[lo:hi]
            test_cluster = target_cluster.get(target, 0)

            # Collect training data from same cluster only
            train_indices = []
            for j, other_target in enumerate(TARGETS):
                if j == i:
                    continue
                other_cluster = target_cluster.get(other_target, -1)
                if other_cluster == test_cluster:
                    olo, ohi = OFFSETS[j], OFFSETS[j + 1]
                    train_indices.extend(range(olo, ohi))

            if len(train_indices) < 100:
                # Fallback: too few in-cluster targets, use all training data
                train_mask = np.ones(len(Y), dtype=bool)
                train_mask[lo:hi] = False
                X_train, y_train = X[train_mask], Y[train_mask]
            else:
                X_train = X[train_indices]
                y_train = Y[train_indices]

            model = HistGradientBoostingRegressor(**model_kwargs)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            m = eval_metrics_for_target(preds, y_test, g_test, k=3)
            all_ndcgs.append(m["ndcg"])
            all_hits.append(m["hit_rate"])
            all_rhos.append(m["spearman"])

        elapsed = time.perf_counter() - t0
        res = {
            "ndcg": float(np.mean(all_ndcgs)),
            "hit_rate": float(np.mean(all_hits)),
            "spearman": float(np.mean(all_rhos)),
            "ndcg_std": float(np.std(all_ndcgs)),
        }
        results[f"within_cluster_k{k}"] = res

        print(f"  B. Within-cluster models (k={k})                         "
              f"  NDCG@3={res['ndcg']:.4f}  "
              f"Hit@1={res['hit_rate']:.3f}  "
              f"Spearman={res['spearman']:.4f}  [{elapsed:.0f}s]")

    # ── Experiment C: SAR profile vector as direct features ──────────────
    print("\n--- Experiment C: SAR profile vector as direct features ---")

    # Use the full SAR profile vector (28 dims) as features
    # alongside the original 11 position features
    profile_df_aligned = pd.DataFrame(index=targets_from_profiles)
    # We need to pass profile_df from the caller — reconstruct from saved CSV
    profile_csv = output_dir / "sar_profiles.csv"
    if profile_csv.exists():
        profile_full = pd.read_csv(profile_csv, index_col=0)
        profile_mat = profile_full.values.astype(np.float32)

        # Build profile feature columns for each row in position data
        profile_features = np.zeros(
            (len(Y), profile_mat.shape[1]), dtype=np.float32,
        )
        for i, target in enumerate(TARGETS):
            lo, hi = OFFSETS[i], OFFSETS[i + 1]
            prof_idx = target_to_idx.get(target)
            if prof_idx is not None:
                profile_features[lo:hi] = profile_mat[prof_idx]

        X_aug = np.column_stack([X, profile_features])
        res = leave_one_target_out(
            X_aug, Y, OFFSETS, GROUPS, TARGETS,
            label=f"C. Global + SAR profile ({X_aug.shape[1]} feat)",
        )
        results["global_sar_profile"] = res

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    t_start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 78)
    print("M7b: Pharmacophore Homology Grouping")
    print("=" * 78)
    print(f"\nBaseline (M7a): NDCG@3=0.964, Hit@1=57%, Spearman=0.607")
    print(f"Question: Does grouping targets by SAR-profile similarity help?\n")

    # ── Step 1: Build SAR profile vectors ────────────────────────────────
    print("\n--- Step 1: Building SAR profile vectors ---")
    logger.info("Loading MMP data from %s ...", MMPS_PATH)
    mmps = pd.read_parquet(
        MMPS_PATH,
        columns=[
            "target_chembl_id", "rgroup_from", "rgroup_to",
            "abs_delta_pActivity",
        ],
    )
    logger.info("  %s MMPs loaded", f"{len(mmps):,}")

    logger.info("Loading R-group properties from %s ...", RGROUP_PROPS_PATH)
    rg_props = pd.read_parquet(RGROUP_PROPS_PATH)
    logger.info("  %s R-groups loaded", f"{len(rg_props):,}")

    profile_df, targets = build_sar_profiles(mmps, rg_props)
    del mmps  # free memory

    # Save profiles
    profile_df.to_csv(OUTPUT_DIR / "sar_profiles.csv")

    # Print sample profiles
    print(f"\n  SAR profile matrix: {profile_df.shape}")
    print(f"  Profile columns: {list(profile_df.columns)}")
    print(f"\n  Sample profiles (first 3 targets, first 6 dims):")
    print(profile_df.iloc[:3, :6].to_string())

    # ── Step 2: Correlation heatmap ──────────────────────────────────────
    print("\n--- Step 2: Target-target SAR profile correlations ---")
    corr_vals = compute_correlation_heatmap(profile_df, OUTPUT_DIR)

    # ── Step 3: Clustering ───────────────────────────────────────────────
    print("\n--- Step 3: Hierarchical clustering ---")
    cluster_assignments = cluster_targets(
        corr_vals, targets, OUTPUT_DIR, n_clusters_range=(5, 8),
    )

    # ── Step 4+5: Evaluation ─────────────────────────────────────────────
    print("\n--- Steps 4-5: Evaluating cluster-conditioned models ---")
    results = evaluate_cluster_conditioned(
        POSITION_DATA_PATH, cluster_assignments, targets, OUTPUT_DIR,
    )

    # ── Save all results ─────────────────────────────────────────────────
    # Convert numpy types for JSON serialization
    def _to_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable_results = {}
    for key, val in results.items():
        if isinstance(val, dict):
            serializable_results[key] = {
                k: _to_serializable(v) for k, v in val.items()
            }
        else:
            serializable_results[key] = _to_serializable(val)

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(serializable_results, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"\nBaseline (M7a):  NDCG@3=0.964  Hit@1=0.570  Spearman=0.607")
    if "baseline" in results:
        b = results["baseline"]
        print(f"Reproduced:      NDCG@3={b['ndcg']:.4f}  "
              f"Hit@1={b['hit_rate']:.3f}  Spearman={b['spearman']:.4f}")

    print("\nBest cluster-conditioned results:")
    best_key, best_ndcg = "", 0.0
    for key, res in results.items():
        if key == "baseline":
            continue
        if res["ndcg"] > best_ndcg:
            best_ndcg = res["ndcg"]
            best_key = key
    if best_key:
        b = results[best_key]
        print(f"  {best_key}: NDCG@3={b['ndcg']:.4f}  "
              f"Hit@1={b['hit_rate']:.3f}  Spearman={b['spearman']:.4f}")

    delta = best_ndcg - results.get("baseline", {}).get("ndcg", 0.964)
    if delta > 0.005:
        print(f"\n  -> Cluster conditioning IMPROVES predictions by {delta:.4f}")
    elif delta < -0.005:
        print(f"\n  -> Cluster conditioning HURTS predictions by {abs(delta):.4f}")
    else:
        print(f"\n  -> Cluster conditioning has NO significant effect (delta={delta:+.4f})")

    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print(f"Total time: {elapsed:.0f}s")
    print("=" * 78)


if __name__ == "__main__":
    main()
