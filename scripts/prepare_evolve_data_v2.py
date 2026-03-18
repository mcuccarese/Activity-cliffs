"""
Prepare evaluation data v2 with RICHER features for ShinkaEvolve.

Key additions over v1:
  - R-group FP XOR bits (256-dim): captures WHICH structural motifs change
  - PCA of XOR bits (top 20 components): compressed version for evolution
  - Functional group change flags (12-dim): adds_halogen, adds_amine, etc.
  - Transform frequency: how common is this R-group swap across ChEMBL
  - R-group size ratio: relative size of outgoing vs incoming R-group

These new features have HIGH within-group variance (different transforms from
the same starting molecule change different structural motifs), unlike env_hash
features which collapse within groups.

Usage:
    python scripts/prepare_evolve_data_v2.py
    python scripts/prepare_evolve_data_v2.py --max-mols-per-target 200
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from rdkit import Chem
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer()

# Popcount lookup for Tanimoto
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)

# SMARTS for functional group classification
_FG_SMARTS = {
    "halogen":     "[F,Cl,Br,I]",
    "amine":       "[NH2,NH1,NH0;!$(NC=O)]",
    "hydroxyl":    "[OH1;!$(OC=O)]",
    "carboxyl":    "C(=O)[OH]",
    "amide":       "C(=O)[NH]",
    "sulfonamide": "S(=O)(=O)N",
    "nitro":       "[N+](=O)[O-]",
    "nitrile":     "C#N",
    "aromatic":    "a",
    "methyl":      "[CH3;!$(C=*)]",
    "ether":       "[OD2;!$(OC=O);!$(O[#1])]",
    "carbonyl":    "[CX3]=[OX1]",
}

# Pre-compile SMARTS patterns
_FG_PATTERNS = {}
for name, smarts in _FG_SMARTS.items():
    pat = Chem.MolFromSmarts(smarts)
    if pat is not None:
        _FG_PATTERNS[name] = pat


def _compute_tanimoto(fp_from: np.ndarray, fp_to: np.ndarray) -> np.ndarray:
    """Vectorised Tanimoto between packed uint8 FP arrays, shape (N, 32)."""
    and_bits = _POPCOUNT[fp_from & fp_to].sum(axis=1).astype(np.float32)
    or_bits = _POPCOUNT[fp_from | fp_to].sum(axis=1).astype(np.float32)
    return np.where(or_bits > 0, and_bits / or_bits, 0.0).astype(np.float32)


def _unpack_fps(fp_bytes_col: pd.Series) -> np.ndarray:
    """Unpack column of 32-byte FP blobs to (N, 256) binary array."""
    packed = np.array(
        [np.frombuffer(b, dtype=np.uint8) for b in fp_bytes_col], dtype=np.uint8
    )  # (N, 32)
    return np.unpackbits(packed, axis=1)  # (N, 256)


def _compute_fg_flags(smiles_series: pd.Series) -> np.ndarray:
    """
    For each SMILES, compute binary flags for functional group presence.
    Returns (N, len(_FG_PATTERNS)) uint8 array.
    """
    n = len(smiles_series)
    n_fg = len(_FG_PATTERNS)
    flags = np.zeros((n, n_fg), dtype=np.uint8)
    fg_names = list(_FG_PATTERNS.keys())

    for i, smi in enumerate(smiles_series):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for j, name in enumerate(fg_names):
            if mol.HasSubstructMatch(_FG_PATTERNS[name]):
                flags[i, j] = 1

    return flags, fg_names


def _compute_rgroup_hac(smiles_series: pd.Series) -> np.ndarray:
    """Compute heavy atom count for each R-group SMILES."""
    hac = np.zeros(len(smiles_series), dtype=np.float32)
    for i, smi in enumerate(smiles_series):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            hac[i] = mol.GetNumHeavyAtoms()
    return hac


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
        Path("evolve/eval_data/eval_data_v2.npz"),
        help="Output .npz file",
    ),
    max_mols_per_target: int = typer.Option(
        200, help="Max starting molecules to sample per target",
    ),
    min_transforms: int = typer.Option(
        5, help="Min transformations per mol_from to include",
    ),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:
    """Prepare enriched evaluation data for ShinkaEvolve v2."""
    rng = np.random.RandomState(seed)

    # ── Step 1: Load MMP identifiers + SMILES ────────────────────────────
    logger.info("Loading MMP identifiers from %s ...", mmps_path)
    mmps = pd.read_parquet(
        mmps_path,
        columns=["target_chembl_id", "mol_from", "abs_delta_pActivity",
                 "rgroup_from", "rgroup_to"],
    )
    logger.info("  %d rows, %d targets", len(mmps), mmps["target_chembl_id"].nunique())

    # ── Step 2: Subsample ─────────────────────────────────────────────────
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

    # ── Step 3: Load delta-descriptors + FP bytes ─────────────────────────
    logger.info("Loading features ...")
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
    logger.info("  Features subsampled: %d rows", len(feats_sub))

    # Also get the R-group SMILES for functional group analysis
    mmps_sub = mmps.iloc[all_indices].reset_index(drop=True)

    # ── Step 4: Compute original features ─────────────────────────────────
    logger.info("Computing Tanimoto ...")
    fp_from_packed = np.array(
        [np.frombuffer(b, dtype=np.uint8) for b in feats_sub["fp_rgroup_from"]],
        dtype=np.uint8,
    )
    fp_to_packed = np.array(
        [np.frombuffer(b, dtype=np.uint8) for b in feats_sub["fp_rgroup_to"]],
        dtype=np.uint8,
    )
    tanimoto = _compute_tanimoto(fp_from_packed, fp_to_packed)

    # ── Step 5: NEW - FP XOR bits ─────────────────────────────────────────
    logger.info("Computing FP XOR bits (256-dim) ...")
    fp_from_bits = np.unpackbits(fp_from_packed, axis=1)  # (N, 256)
    fp_to_bits = np.unpackbits(fp_to_packed, axis=1)      # (N, 256)
    fp_xor = (fp_from_bits ^ fp_to_bits).astype(np.float32)  # (N, 256)
    logger.info("  XOR bits: mean bits changed per transform: %.1f / 256",
                fp_xor.sum(axis=1).mean())

    # PCA of XOR bits -> top 20 components
    logger.info("PCA of XOR bits -> 20 components ...")
    pca = PCA(n_components=20, random_state=42)
    fp_xor_pca = pca.fit_transform(fp_xor).astype(np.float32)
    logger.info("  PCA explained variance: %.1f%%", pca.explained_variance_ratio_.sum() * 100)

    # ── Step 6: NEW - Functional group change flags ───────────────────────
    logger.info("Computing functional group flags for R-groups ...")
    logger.info("  Processing rgroup_from ...")
    fg_from, fg_names = _compute_fg_flags(mmps_sub["rgroup_from"])
    logger.info("  Processing rgroup_to ...")
    fg_to, _ = _compute_fg_flags(mmps_sub["rgroup_to"])

    # Change flags: gained (0->1) and lost (1->0) for each FG
    fg_gained = ((fg_from == 0) & (fg_to == 1)).astype(np.float32)  # (N, 12)
    fg_lost = ((fg_from == 1) & (fg_to == 0)).astype(np.float32)    # (N, 12)
    # Net change: +1 = gained, -1 = lost, 0 = unchanged
    fg_net = (fg_gained - fg_lost).astype(np.float32)  # (N, 12)
    logger.info("  FG names: %s", fg_names)
    logger.info("  Mean FGs gained per transform: %.2f", fg_gained.sum(axis=1).mean())
    logger.info("  Mean FGs lost per transform: %.2f", fg_lost.sum(axis=1).mean())

    # ── Step 7: NEW - Transform frequency ─────────────────────────────────
    logger.info("Computing transform frequency ...")
    # Create canonical transform key (sorted pair of R-group SMILES)
    rg_pairs = mmps_sub.apply(
        lambda r: tuple(sorted([r["rgroup_from"], r["rgroup_to"]])), axis=1
    )
    pair_counts = rg_pairs.value_counts()
    transform_freq = rg_pairs.map(pair_counts).values.astype(np.float32)
    # Log-transform (frequencies are very skewed)
    transform_log_freq = np.log1p(transform_freq).astype(np.float32)
    logger.info("  Transform freq: mean=%.1f, median=%.1f, max=%.0f",
                transform_freq.mean(), np.median(transform_freq), transform_freq.max())

    # ── Step 8: NEW - R-group size ratio ──────────────────────────────────
    logger.info("Computing R-group size features ...")
    hac_from = _compute_rgroup_hac(mmps_sub["rgroup_from"])
    hac_to = _compute_rgroup_hac(mmps_sub["rgroup_to"])
    # Size ratio (log scale, handles zeros)
    size_ratio = np.log1p(hac_to) - np.log1p(hac_from)
    # Max of the two (indicates complexity of the more complex R-group)
    size_max = np.maximum(hac_from, hac_to)

    # ── Step 9: Assemble feature matrix ───────────────────────────────────
    logger.info("Assembling feature matrix ...")

    # Original 8 features (indices 0-7)
    X_orig = np.column_stack([
        feats_sub[delta_cols].values.astype(np.float32),  # 0-6: delta descriptors
        tanimoto.reshape(-1, 1),                           # 7: FP Tanimoto
    ])

    # New features
    X_new = np.column_stack([
        fp_xor_pca,                    # 8-27: PCA of FP XOR bits (20 dims)
        fg_net,                        # 28-39: FG net change flags (12 dims)
        transform_log_freq.reshape(-1, 1),  # 40: log transform frequency
        size_ratio.reshape(-1, 1),     # 41: R-group size ratio
        size_max.reshape(-1, 1),       # 42: max R-group size
    ])

    X_all = np.column_stack([X_orig, X_new]).astype(np.float32)

    y = mmps_sub["abs_delta_pActivity"].values.astype(np.float32)
    groups = mmps_sub["mol_from"].values.astype(np.int64)

    # Also save the full XOR bits separately (for ML ceiling test)
    logger.info("  X shape: %s (orig 8 + new %d)", X_all.shape, X_new.shape[1])

    # ── Step 10: Feature name manifest ────────────────────────────────────
    feature_names = (
        delta_cols + ["fp_tanimoto"]  # 0-7
        + [f"xor_pc{i}" for i in range(20)]  # 8-27
        + [f"fg_net_{name}" for name in fg_names]  # 28-39
        + ["transform_log_freq", "size_ratio", "size_max"]  # 40-42
    )

    # ── Step 11: Save ─────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=X_all,
        y=y,
        groups=groups,
        target_offsets=np.array(target_offsets, dtype=np.int64),
        target_names=np.array(target_names),
        feature_names=np.array(feature_names),
        # Also save full XOR bits for ML ceiling
        fp_xor=fp_xor,
        # PCA model info
        pca_components=pca.components_.astype(np.float32),
        pca_mean=pca.mean_.astype(np.float32),
    )

    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info("Saved %s (%.1f MB)", output_path, size_mb)
    logger.info("  X shape: %s", X_all.shape)
    logger.info("  Features: %s", feature_names)
    logger.info("  %d targets, %d total rows", len(target_names), len(y))


if __name__ == "__main__":
    app()
