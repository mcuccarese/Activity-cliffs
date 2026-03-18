"""
Prepare evaluation data for ShinkaEvolve scoring-function evolution.

Reads all_mmps.parquet + mmp_features.parquet, subsamples for fast fitness
evaluation, computes FP Tanimoto, and saves a compact .npz file.

Usage:
    python scripts/prepare_evolve_data.py
    python scripts/prepare_evolve_data.py --max-mols-per-target 300 --min-transforms 3
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import typer

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer()

# Popcount lookup table for Tanimoto on packed uint8 fingerprints
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)


def _compute_tanimoto(fp_from_col: pd.Series, fp_to_col: pd.Series) -> np.ndarray:
    """Vectorised Tanimoto between two columns of 32-byte FP blobs."""
    fp_from = np.array(
        [np.frombuffer(b, dtype=np.uint8) for b in fp_from_col], dtype=np.uint8
    )  # (N, 32)
    fp_to = np.array(
        [np.frombuffer(b, dtype=np.uint8) for b in fp_to_col], dtype=np.uint8
    )  # (N, 32)

    and_bits = _POPCOUNT[fp_from & fp_to].sum(axis=1).astype(np.float32)
    or_bits = _POPCOUNT[fp_from | fp_to].sum(axis=1).astype(np.float32)
    return np.where(or_bits > 0, and_bits / or_bits, 0.0).astype(np.float32)


@app.command()
def main(
    mmps_path: Path = typer.Option(
        Path("outputs/mmps/all_mmps.parquet"),
        help="Path to all_mmps.parquet",
    ),
    features_path: Path = typer.Option(
        Path("outputs/features/mmp_features.parquet"),
        help="Path to mmp_features.parquet",
    ),
    output_path: Path = typer.Option(
        Path("evolve/eval_data/eval_data.npz"),
        help="Output .npz file for ShinkaEvolve evaluation",
    ),
    max_mols_per_target: int = typer.Option(
        200, help="Max starting molecules to sample per target",
    ),
    min_transforms: int = typer.Option(
        5, help="Min transformations per mol_from to include",
    ),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
) -> None:
    """Prepare subsampled evaluation data for ShinkaEvolve."""
    rng = np.random.RandomState(seed)

    # ── Step 1: Load MMP identifiers ──────────────────────────────────────────
    logger.info("Loading MMP identifiers from %s ...", mmps_path)
    mmps = pd.read_parquet(
        mmps_path,
        columns=["target_chembl_id", "mol_from", "abs_delta_pActivity"],
    )
    logger.info("  %d rows, %d targets", len(mmps), mmps["target_chembl_id"].nunique())

    # ── Step 2: Identify eligible mol_from per target ─────────────────────────
    logger.info("Subsampling: min_transforms=%d, max_mols_per_target=%d ...",
                min_transforms, max_mols_per_target)

    keep_indices: list[np.ndarray] = []
    target_names: list[str] = []
    target_offsets: list[int] = [0]

    for target, tdf in mmps.groupby("target_chembl_id"):
        # Count transforms per mol_from
        mol_counts = tdf["mol_from"].value_counts()
        eligible = mol_counts[mol_counts >= min_transforms].index.values

        if len(eligible) == 0:
            continue

        # Sample mol_from values
        if len(eligible) > max_mols_per_target:
            chosen = rng.choice(eligible, size=max_mols_per_target, replace=False)
        else:
            chosen = eligible

        # Get row indices for chosen mol_from values
        mask = tdf["mol_from"].isin(chosen)
        idx = tdf.index[mask].values
        keep_indices.append(idx)
        target_names.append(str(target))
        target_offsets.append(target_offsets[-1] + len(idx))

    all_indices = np.concatenate(keep_indices)
    logger.info("  Selected %d rows across %d targets", len(all_indices), len(target_names))

    # ── Step 3: Load features for selected rows ──────────────────────────────
    logger.info("Loading features ...")

    # Read delta-descriptor columns (light, float32)
    delta_cols = [
        "delta_MW", "delta_LogP", "delta_TPSA",
        "delta_HBDonors", "delta_HBAcceptors",
        "delta_RotBonds", "delta_HeavyAtomCount",
    ]
    feats = pd.read_parquet(features_path, columns=delta_cols + ["fp_rgroup_from", "fp_rgroup_to"])
    feats_sub = feats.iloc[all_indices].reset_index(drop=True)
    del feats
    logger.info("  Features subsampled: %d rows", len(feats_sub))

    # ── Step 4: Compute FP Tanimoto ───────────────────────────────────────────
    logger.info("Computing FP Tanimoto ...")
    tanimoto = _compute_tanimoto(feats_sub["fp_rgroup_from"], feats_sub["fp_rgroup_to"])
    logger.info("  Tanimoto stats: mean=%.3f, std=%.3f", tanimoto.mean(), tanimoto.std())

    # ── Step 5: Assemble feature matrix X ─────────────────────────────────────
    X = np.column_stack([
        feats_sub[delta_cols].values.astype(np.float32),
        tanimoto.reshape(-1, 1),
    ])  # (N, 8)
    del feats_sub

    # Ground truth: abs_delta_pActivity
    mmps_sub = mmps.iloc[all_indices].reset_index(drop=True)
    y = mmps_sub["abs_delta_pActivity"].values.astype(np.float32)

    # Group IDs: mol_from (for per-molecule NDCG)
    groups = mmps_sub["mol_from"].values.astype(np.int64)

    # ── Step 6: Save ──────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        groups=groups,
        target_offsets=np.array(target_offsets, dtype=np.int64),
        target_names=np.array(target_names),
    )
    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info("Saved %s (%.1f MB)", output_path, size_mb)
    logger.info("  X shape: %s, y shape: %s, groups shape: %s", X.shape, y.shape, groups.shape)
    logger.info("  %d targets, offsets: %s", len(target_names), target_offsets[:5])


if __name__ == "__main__":
    app()
