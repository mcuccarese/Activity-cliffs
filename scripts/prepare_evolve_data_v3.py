#!/usr/bin/env python
"""
M6c: Prepare eval data with 3D context × change-type interaction features.

Joins:
  - Original 2D delta descriptors (7) + Tanimoto (1)  → 8 features
  - 3D pharmacophore context at attachment (9)         → 9 features
  - Change-type delta vector (11)                      → 11 features
  - Interaction cross-product: context × change_type   → 99 features
                                                  Total: 127 features

Uses the same subsampling protocol as prepare_evolve_data.py (seed=42,
max_mols_per_target=200, min_transforms=5) so results are directly
comparable to the v1/v2 ceiling baselines.

Usage:
    python scripts/prepare_evolve_data_v3.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import typer

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)
sys.stdout.reconfigure(line_buffering=True)

app = typer.Typer()

# Popcount lookup table for Tanimoto on packed uint8 fingerprints
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)


def _compute_tanimoto(fp_from_col: pd.Series, fp_to_col: pd.Series) -> np.ndarray:
    """Vectorised Tanimoto between two columns of 32-byte FP blobs."""
    fp_from = np.array(
        [np.frombuffer(b, dtype=np.uint8) for b in fp_from_col], dtype=np.uint8
    )
    fp_to = np.array(
        [np.frombuffer(b, dtype=np.uint8) for b in fp_to_col], dtype=np.uint8
    )
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
    context_3d_path: Path = typer.Option(
        Path("outputs/features/context_3d.parquet"),
        help="Path to context_3d.parquet (M6a output)",
    ),
    rgroup_props_path: Path = typer.Option(
        Path("outputs/features/rgroup_props.parquet"),
        help="Path to rgroup_props.parquet (M6b output)",
    ),
    output_path: Path = typer.Option(
        Path("evolve/eval_data/eval_data_v3.npz"),
        help="Output .npz file",
    ),
    max_mols_per_target: int = typer.Option(
        200, help="Max starting molecules per target (same as v1)",
    ),
    min_transforms: int = typer.Option(
        5, help="Min transformations per mol_from (same as v1)",
    ),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:
    """Prepare eval data v3: 2D deltas + 3D context + change-type + interactions."""
    t_start = time.perf_counter()
    rng = np.random.RandomState(seed)

    # ── Step 1: Load MMP identifiers ──────────────────────────────────────────
    logger.info("Loading MMP identifiers from %s ...", mmps_path)
    mmps = pd.read_parquet(
        mmps_path,
        columns=[
            "target_chembl_id", "mol_from", "abs_delta_pActivity",
            "core_smiles", "rgroup_from", "rgroup_to",
        ],
    )
    logger.info("  %d rows, %d targets", len(mmps), mmps["target_chembl_id"].nunique())

    # ── Step 2: Subsample (same protocol as v1) ──────────────────────────────
    logger.info("Subsampling: min_transforms=%d, max_mols_per_target=%d ...",
                min_transforms, max_mols_per_target)

    keep_indices: list[np.ndarray] = []
    target_names: list[str] = []
    target_offsets: list[int] = [0]

    for target, tdf in mmps.groupby("target_chembl_id"):
        mol_counts = tdf["mol_from"].value_counts()
        eligible = mol_counts[mol_counts >= min_transforms].index.values
        if len(eligible) == 0:
            continue
        if len(eligible) > max_mols_per_target:
            chosen = rng.choice(eligible, size=max_mols_per_target, replace=False)
        else:
            chosen = eligible
        mask = tdf["mol_from"].isin(chosen)
        idx = tdf.index[mask].values
        keep_indices.append(idx)
        target_names.append(str(target))
        target_offsets.append(target_offsets[-1] + len(idx))

    all_indices = np.concatenate(keep_indices)
    logger.info("  Selected %d rows across %d targets", len(all_indices), len(target_names))

    # Subsample the MMP table
    mmps_sub = mmps.iloc[all_indices].reset_index(drop=True)
    del mmps

    # ── Step 3: Load original delta features + FP for tanimoto ────────────────
    logger.info("Loading 2D delta features ...")
    delta_cols = [
        "delta_MW", "delta_LogP", "delta_TPSA",
        "delta_HBDonors", "delta_HBAcceptors",
        "delta_RotBonds", "delta_HeavyAtomCount",
    ]
    feats = pd.read_parquet(
        features_path,
        columns=delta_cols + ["fp_rgroup_from", "fp_rgroup_to"],
    )
    feats_sub = feats.iloc[all_indices].reset_index(drop=True)
    del feats

    logger.info("Computing FP Tanimoto ...")
    tanimoto = _compute_tanimoto(feats_sub["fp_rgroup_from"], feats_sub["fp_rgroup_to"])

    X_deltas = np.column_stack([
        feats_sub[delta_cols].values.astype(np.float32),
        tanimoto.reshape(-1, 1),
    ])  # (N, 8)
    del feats_sub, tanimoto
    logger.info("  2D delta features: shape %s", X_deltas.shape)

    # ── Step 4: Load 3D context lookup and join on core_smiles ────────────────
    logger.info("Loading 3D context features from %s ...", context_3d_path)
    ctx_df = pd.read_parquet(context_3d_path)
    ctx_cols = [c for c in ctx_df.columns if c != "core_smiles"]
    ctx_lookup = dict(zip(ctx_df["core_smiles"], ctx_df[ctx_cols].values.astype(np.float32)))
    n_ctx = len(ctx_cols)
    logger.info("  %d cores in lookup, %d features: %s", len(ctx_lookup), n_ctx, ctx_cols)

    # Map context features for subsampled rows
    zero_ctx = np.zeros(n_ctx, dtype=np.float32)
    ctx_list = []
    n_miss_ctx = 0
    for smi in mmps_sub["core_smiles"]:
        row = ctx_lookup.get(smi)
        if row is not None:
            ctx_list.append(row)
        else:
            ctx_list.append(zero_ctx)
            n_miss_ctx += 1
    X_context = np.array(ctx_list, dtype=np.float32)
    del ctx_lookup, ctx_list
    logger.info("  3D context joined: shape %s (%d missing → zero-filled)", X_context.shape, n_miss_ctx)

    # ── Step 5: Load R-group property lookup and compute change-type deltas ───
    logger.info("Loading R-group properties from %s ...", rgroup_props_path)
    rg_df = pd.read_parquet(rgroup_props_path)
    prop_cols = [c for c in rg_df.columns if c != "rgroup_smiles"]
    rg_lookup = dict(zip(rg_df["rgroup_smiles"], rg_df[prop_cols].values.astype(np.float32)))
    n_props = len(prop_cols)
    logger.info("  %d R-groups in lookup, %d properties: %s", len(rg_lookup), n_props, prop_cols)

    zero_props = np.zeros(n_props, dtype=np.float32)
    props_from_list = []
    props_to_list = []
    n_miss_rg = 0
    for rg_from, rg_to in zip(mmps_sub["rgroup_from"], mmps_sub["rgroup_to"]):
        pf = rg_lookup.get(rg_from)
        pt = rg_lookup.get(rg_to)
        if pf is None:
            pf = zero_props
            n_miss_rg += 1
        if pt is None:
            pt = zero_props
            n_miss_rg += 1
        props_from_list.append(pf)
        props_to_list.append(pt)

    props_from = np.array(props_from_list, dtype=np.float32)
    props_to = np.array(props_to_list, dtype=np.float32)
    X_change_type = props_to - props_from  # (N, 11) — the delta vector
    del rg_lookup, props_from_list, props_to_list, props_from, props_to
    logger.info("  Change-type deltas: shape %s (%d missing R-groups → zero-filled)",
                X_change_type.shape, n_miss_rg)

    # ── Step 6: Compute interaction features (outer product) ──────────────────
    logger.info("Computing context × change_type interactions ...")
    # Outer product: (N, 9) × (N, 11) → (N, 9*11=99)
    # For each row: context[i] ⊗ change_type[i] = 99-dim vector
    X_interact = (
        X_context[:, :, np.newaxis] * X_change_type[:, np.newaxis, :]
    ).reshape(len(X_context), -1).astype(np.float32)

    # Name the interaction features
    interact_names = []
    for c in ctx_cols:
        for p in prop_cols:
            interact_names.append(f"{c}_x_delta_{p}")

    logger.info("  Interaction features: shape %s", X_interact.shape)

    # ── Step 7: Assemble and save ─────────────────────────────────────────────
    delta_feat_names = delta_cols + ["tanimoto"]
    ctx_feat_names = [f"ctx_{c}" for c in ctx_cols]
    ct_feat_names = [f"delta_{p}" for p in prop_cols]

    all_feat_names = delta_feat_names + ctx_feat_names + ct_feat_names + interact_names
    X_all = np.column_stack([X_deltas, X_context, X_change_type, X_interact])

    y = mmps_sub["abs_delta_pActivity"].values.astype(np.float32)
    groups = mmps_sub["mol_from"].values.astype(np.int64)

    logger.info("Final feature matrix: %s (%d features)", X_all.shape, len(all_feat_names))
    logger.info("  2D deltas:     cols 0-%d", len(delta_feat_names) - 1)
    logger.info("  3D context:    cols %d-%d", len(delta_feat_names),
                len(delta_feat_names) + len(ctx_feat_names) - 1)
    logger.info("  Change-type:   cols %d-%d",
                len(delta_feat_names) + len(ctx_feat_names),
                len(delta_feat_names) + len(ctx_feat_names) + len(ct_feat_names) - 1)
    logger.info("  Interactions:  cols %d-%d",
                len(delta_feat_names) + len(ctx_feat_names) + len(ct_feat_names),
                len(all_feat_names) - 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=X_all,
        y=y,
        groups=groups,
        target_offsets=np.array(target_offsets, dtype=np.int64),
        target_names=np.array(target_names),
        feature_names=np.array(all_feat_names),
    )
    size_mb = output_path.stat().st_size / 1024 / 1024
    elapsed = time.perf_counter() - t_start
    logger.info("Saved %s (%.1f MB) in %.0f s", output_path, size_mb, elapsed)
    logger.info("  X: %s, y: %s, groups: %s, %d targets",
                X_all.shape, y.shape, groups.shape, len(target_names))


if __name__ == "__main__":
    app()
