#!/usr/bin/env python
"""
OOD Experiment 5: Feature Robustness / Sensitivity Analysis

Tests sensitivity to arbitrary choices in feature computation:
  5a: Pharmacophore radius — recompute 3D context features at radii
      3.0, 3.5, 4.0 (baseline), 4.5, 5.0, 6.0 A for 2,000 sampled cores.
      Rebuild position-level data for those cores, retrain HGB with
      LOO-target evaluation, report NDCG@3 at each radius.
  5b: Conformer ensemble — generate 10 ETKDG conformers for 1,000 cores,
      compute features on each, report coefficient of variation per feature
      and fraction of position rankings that change.

Usage:
    python scripts/ood/feature_sensitivity.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdFreeSASA
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

from activity_cliffs.features.context_3d import (
    _find_attachment,
    _prepare_for_embedding,
    _classify_pharmacophore,
    _rotatable_bonds_near,
)

sys.stdout.reconfigure(line_buffering=True)


# ── 3D context computation with configurable radius ─────────────────────────

def compute_3d_context_at_radius(
    core_smi: str, radius: float = 4.0, seed: int = 42,
) -> np.ndarray:
    """Compute 3D pharmacophore context features with configurable radius.

    Same algorithm as context_3d.compute_3d_context but the pharmacophore
    distance cutoff is parameterised instead of hard-coded at 4.0 A.

    Returns float32 ndarray of shape (9,).
    """
    ZERO = np.zeros(9, dtype=np.float32)
    mol = Chem.MolFromSmiles(core_smi)
    if mol is None:
        return ZERO

    dummy_idx, attach_idx = _find_attachment(mol)
    if attach_idx is None:
        return ZERO

    # Topological features (radius-independent)
    is_aromatic = float(mol.GetAtomWithIdx(attach_idx).GetIsAromatic())
    n_rotbonds = float(_rotatable_bonds_near(mol, attach_idx, max_bond_dist=2))

    # Gasteiger charge (radius-independent)
    try:
        mol_capped = Chem.RWMol(mol)
        mol_capped.GetAtomWithIdx(dummy_idx).SetAtomicNum(1)
        mol_capped.GetAtomWithIdx(dummy_idx).SetAtomMapNum(0)
        mol_capped.GetAtomWithIdx(dummy_idx).SetIsotope(0)
        mol_capped = mol_capped.GetMol()
        AllChem.ComputeGasteigerCharges(mol_capped)
        charge = float(
            mol_capped.GetAtomWithIdx(attach_idx).GetDoubleProp("_GasteigerCharge")
        )
        if not np.isfinite(charge):
            charge = 0.0
    except Exception:
        charge = 0.0

    # 3D embedding
    mol3d = _prepare_for_embedding(mol, dummy_idx)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.maxIterations = 200

    if AllChem.EmbedMolecule(mol3d, params) != 0:
        if AllChem.EmbedMolecule(mol3d, randomSeed=seed) != 0:
            feat = ZERO.copy()
            feat[5], feat[6], feat[7] = charge, n_rotbonds, is_aromatic
            return feat

    try:
        AllChem.MMFFOptimizeMolecule(mol3d, maxIters=200)
    except Exception:
        pass

    conf = mol3d.GetConformer()
    attach_pos = np.array(conf.GetAtomPosition(attach_idx))

    # Pharmacophore counts within configurable radius
    n_donor = n_acceptor = n_hydrophobic = n_aromatic_r = n_heavy = 0
    for atom in mol3d.GetAtoms():
        idx = atom.GetIdx()
        if idx == attach_idx or atom.GetAtomicNum() <= 1:
            continue
        pos = np.array(conf.GetAtomPosition(idx))
        if np.linalg.norm(pos - attach_pos) > radius:
            continue
        n_heavy += 1
        ptype = _classify_pharmacophore(atom)
        if ptype == "donor":
            n_donor += 1
        elif ptype == "acceptor":
            n_acceptor += 1
        elif ptype == "hydrophobic":
            n_hydrophobic += 1
        elif ptype == "aromatic":
            n_aromatic_r += 1

    # SASA (radius-independent)
    try:
        radii_sasa = rdFreeSASA.classifyAtoms(mol3d)
        rdFreeSASA.CalcSASA(mol3d, radii_sasa)
        sasa = float(mol3d.GetAtomWithIdx(attach_idx).GetDoubleProp("SASA"))
    except Exception:
        sasa = 0.0

    return np.array(
        [n_donor, n_acceptor, n_hydrophobic, n_aromatic_r,
         sasa, charge, n_rotbonds, is_aromatic, n_heavy],
        dtype=np.float32,
    )


def compute_3d_context_multi_conf(
    core_smi: str, n_confs: int = 10, radius: float = 4.0,
) -> np.ndarray:
    """Compute 3D context features across multiple ETKDG conformers.

    Returns float32 ndarray of shape (n_successful, 9).
    Falls back to shape (1, 9) if embedding fails entirely.
    """
    ZERO = np.zeros((1, 9), dtype=np.float32)
    mol = Chem.MolFromSmiles(core_smi)
    if mol is None:
        return ZERO

    dummy_idx, attach_idx = _find_attachment(mol)
    if attach_idx is None:
        return ZERO

    is_aromatic = float(mol.GetAtomWithIdx(attach_idx).GetIsAromatic())
    n_rotbonds = float(_rotatable_bonds_near(mol, attach_idx, max_bond_dist=2))

    # Gasteiger charge (conformer-independent)
    try:
        mol_capped = Chem.RWMol(mol)
        mol_capped.GetAtomWithIdx(dummy_idx).SetAtomicNum(1)
        mol_capped.GetAtomWithIdx(dummy_idx).SetAtomMapNum(0)
        mol_capped.GetAtomWithIdx(dummy_idx).SetIsotope(0)
        mol_capped = mol_capped.GetMol()
        AllChem.ComputeGasteigerCharges(mol_capped)
        charge = float(
            mol_capped.GetAtomWithIdx(attach_idx).GetDoubleProp("_GasteigerCharge")
        )
        if not np.isfinite(charge):
            charge = 0.0
    except Exception:
        charge = 0.0

    # Generate multiple conformers
    mol3d = _prepare_for_embedding(mol, dummy_idx)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.maxIterations = 200
    cids = list(AllChem.EmbedMultipleConfs(mol3d, numConfs=n_confs, params=params))
    if len(cids) == 0:
        cids = list(AllChem.EmbedMultipleConfs(mol3d, numConfs=n_confs, randomSeed=42))
    if len(cids) == 0:
        feat = ZERO.copy()
        feat[0, 5], feat[0, 6], feat[0, 7] = charge, n_rotbonds, is_aromatic
        return feat

    # MMFF optimise all conformers
    try:
        AllChem.MMFFOptimizeMoleculeConfs(mol3d, maxIters=200)
    except Exception:
        pass

    # SASA radii (computed once)
    try:
        sasa_radii = rdFreeSASA.classifyAtoms(mol3d)
    except Exception:
        sasa_radii = None

    results = []
    for cid in cids:
        conf = mol3d.GetConformer(cid)
        attach_pos = np.array(conf.GetAtomPosition(attach_idx))

        n_donor = n_acceptor = n_hydrophobic = n_aromatic_r = n_heavy = 0
        for atom in mol3d.GetAtoms():
            idx = atom.GetIdx()
            if idx == attach_idx or atom.GetAtomicNum() <= 1:
                continue
            pos = np.array(conf.GetAtomPosition(idx))
            if np.linalg.norm(pos - attach_pos) > radius:
                continue
            n_heavy += 1
            ptype = _classify_pharmacophore(atom)
            if ptype == "donor":
                n_donor += 1
            elif ptype == "acceptor":
                n_acceptor += 1
            elif ptype == "hydrophobic":
                n_hydrophobic += 1
            elif ptype == "aromatic":
                n_aromatic_r += 1

        sasa = 0.0
        if sasa_radii is not None:
            try:
                rdFreeSASA.CalcSASA(mol3d, sasa_radii, confIdx=cid)
                sasa = float(
                    mol3d.GetAtomWithIdx(attach_idx).GetDoubleProp("SASA")
                )
            except Exception:
                pass

        results.append([
            float(n_donor), float(n_acceptor), float(n_hydrophobic),
            float(n_aromatic_r), sasa, charge, n_rotbonds, is_aromatic,
            float(n_heavy),
        ])

    return np.array(results, dtype=np.float32)


# ── Evaluation helpers ──────────────────────────────────────────────────────

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


def eval_metrics_for_target(scores, y, groups, k=3):
    ndcgs, hits = [], []
    for mol_id in np.unique(groups):
        mask = groups == mol_id
        if mask.sum() < k:
            continue
        ndcgs.append(ndcg_at_k(y[mask], scores[mask], k))
        hits.append(float(np.argmax(scores[mask]) == np.argmax(y[mask])))
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


HGB_KWARGS = {
    "max_iter": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "min_samples_leaf": 50,
    "random_state": 42,
}


# ── Position data builder ──────────────────────────────────────────────────

def _core_n_heavy(smi: str) -> int:
    mol = Chem.MolFromSmiles(smi)
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 0) if mol else 0


def _core_n_rings(smi: str) -> int:
    mol = Chem.MolFromSmiles(smi)
    return mol.GetRingInfo().NumRings() if mol else 0


def build_position_data(mmps: pd.DataFrame, ctx_lookup: dict, core_topo: dict):
    """Aggregate MMPs to position level and build evaluation arrays.

    Parameters
    ----------
    mmps : DataFrame with target_chembl_id, mol_from, core_smiles, abs_delta_pActivity
    ctx_lookup : dict[core_smiles] -> ndarray(9,) of context features
    core_topo : dict[core_smiles] -> (n_heavy, n_rings)

    Returns X, y, groups, offsets, target_names  (or None if too little data).
    """
    pos = (
        mmps
        .groupby(["core_smiles", "target_chembl_id"])
        .agg(sensitivity=("abs_delta_pActivity", "mean"), n_mmps=("abs_delta_pActivity", "count"))
        .reset_index()
    )
    pos = pos[pos["n_mmps"] >= 3].reset_index(drop=True)

    mol_pos = (
        mmps[["mol_from", "core_smiles", "target_chembl_id"]]
        .drop_duplicates()
        .merge(pos, on=["core_smiles", "target_chembl_id"])
    )

    # Keep molecules with >= 3 positions
    cnt = mol_pos.groupby(["mol_from", "target_chembl_id"]).size().reset_index(name="n_pos")
    cnt = cnt[cnt["n_pos"] >= 3]
    mol_pos = mol_pos.merge(cnt[["mol_from", "target_chembl_id"]], on=["mol_from", "target_chembl_id"])

    if len(mol_pos) < 100:
        return None

    mol_pos = mol_pos.sort_values(["target_chembl_id", "mol_from", "core_smiles"]).reset_index(drop=True)

    target_names, target_offsets = [], [0]
    for target, tdf in mol_pos.groupby("target_chembl_id", sort=True):
        target_names.append(str(target))
        target_offsets.append(target_offsets[-1] + len(tdf))

    zero_ctx = np.zeros(9, dtype=np.float32)
    X_rows = []
    for smi in mol_pos["core_smiles"]:
        ctx = ctx_lookup.get(smi, zero_ctx)
        nh, nr = core_topo.get(smi, (0, 0))
        X_rows.append(np.concatenate([ctx, [float(nh), float(nr)]]))

    X = np.array(X_rows, dtype=np.float32)
    y = mol_pos["sensitivity"].values.astype(np.float32)
    groups = mol_pos["mol_from"].values.astype(np.int64)
    offsets = np.array(target_offsets, dtype=np.int64)
    return X, y, groups, offsets, target_names


def loo_target_eval(X, Y, groups, offsets, targets, label):
    """LOO-target HGB evaluation. Returns dict with metrics."""
    all_ndcgs, all_hits, all_rhos = [], [], []
    t0 = time.perf_counter()
    for i in range(len(targets)):
        lo, hi = offsets[i], offsets[i + 1]
        if hi - lo < 10:
            continue
        train_mask = np.ones(len(Y), dtype=bool)
        train_mask[lo:hi] = False
        if train_mask.sum() < 100:
            continue
        model = HistGradientBoostingRegressor(**HGB_KWARGS)
        model.fit(X[train_mask], Y[train_mask])
        preds = model.predict(X[lo:hi])
        m = eval_metrics_for_target(preds, Y[lo:hi], groups[lo:hi])
        if m["n_groups"] > 0:
            all_ndcgs.append(m["ndcg"])
            all_hits.append(m["hit_rate"])
            all_rhos.append(m["spearman"])
    elapsed = time.perf_counter() - t0
    if not all_ndcgs:
        print(f"  {label:45s}  [NO DATA]")
        return None
    result = {
        "label": label,
        "ndcg": float(np.mean(all_ndcgs)),
        "hit_rate": float(np.mean(all_hits)),
        "spearman": float(np.mean(all_rhos)),
        "ndcg_std": float(np.std(all_ndcgs)),
        "n_targets_eval": len(all_ndcgs),
        "elapsed_s": round(elapsed, 1),
    }
    print(f"  {label:45s}  NDCG@3={result['ndcg']:.4f} (+-{result['ndcg_std']:.4f})  "
          f"Hit@1={result['hit_rate']:.3f}  Spearman={result['spearman']:.4f}  "
          f"[{elapsed:.0f}s, {result['n_targets_eval']} targets]")
    return result


# ── Experiment 5a: Pharmacophore radius sensitivity ─────────────────────────

def radius_sensitivity(
    sampled_cores: list[str],
    mmps: pd.DataFrame,
    core_topo: dict[str, tuple[int, int]],
):
    print("\n" + "=" * 78)
    print("EXPERIMENT 5A: PHARMACOPHORE RADIUS SENSITIVITY")
    print("=" * 78)

    radii = [3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
    results = []

    for radius in radii:
        print(f"\n--- Radius = {radius:.1f} A ---")
        t0 = time.perf_counter()

        # Compute 3D context features at this radius
        ctx_lookup = {}
        n_fail = 0
        for j, smi in enumerate(sampled_cores):
            if (j + 1) % 500 == 0:
                print(f"  computing features: {j+1}/{len(sampled_cores)}")
            feat = compute_3d_context_at_radius(smi, radius=radius)
            ctx_lookup[smi] = feat
            if np.all(feat == 0):
                n_fail += 1

        feat_time = time.perf_counter() - t0
        print(f"  Feature computation: {feat_time:.0f}s ({n_fail} failures)")

        # Build position data with these features
        data = build_position_data(mmps, ctx_lookup, core_topo)
        if data is None:
            print(f"  [SKIP] too little position data at radius {radius}")
            continue
        X, y, groups, offsets, targets = data
        print(f"  Position data: {len(y):,} rows, {len(targets)} targets")

        # LOO-target HGB evaluation
        r = loo_target_eval(X, y, groups, offsets, targets,
                            f"HGB (11 feat, radius={radius:.1f})")
        if r:
            r["radius"] = radius
            results.append(r)

        # Also test heuristic for reference
        idx_nh = 9  # core_n_heavy is always column 9
        heur_ndcgs = []
        for i in range(len(targets)):
            lo, hi = offsets[i], offsets[i + 1]
            if hi - lo < 10:
                continue
            scores = -X[lo:hi, idx_nh]
            m = eval_metrics_for_target(scores, y[lo:hi], groups[lo:hi])
            if m["n_groups"] > 0:
                heur_ndcgs.append(m["ndcg"])
        if heur_ndcgs:
            heur_result = {
                "label": f"-core_n_heavy heuristic (radius={radius:.1f})",
                "radius": radius,
                "ndcg": float(np.mean(heur_ndcgs)),
                "ndcg_std": float(np.std(heur_ndcgs)),
            }
            print(f"  {heur_result['label']:45s}  NDCG@3={heur_result['ndcg']:.4f}")

    # Summary
    print("\n--- Radius Sensitivity Summary ---")
    print(f"  {'Radius':>8s}  {'NDCG@3':>8s}  {'Spearman':>10s}")
    print("  " + "-" * 30)
    for r in results:
        print(f"  {r['radius']:>7.1f}A  {r['ndcg']:.4f}    {r['spearman']:.4f}")

    if len(results) >= 2:
        ndcg_range = max(r["ndcg"] for r in results) - min(r["ndcg"] for r in results)
        print(f"\n  NDCG@3 range across radii: {ndcg_range:.4f}")
        if ndcg_range < 0.01:
            print("  -> ROBUST: performance is insensitive to radius choice")
        elif ndcg_range < 0.03:
            print("  -> MODERATE: some sensitivity to radius")
        else:
            print("  -> SENSITIVE: radius choice matters")

    return results


# ── Experiment 5b: Conformer ensemble stability ─────────────────────────────

def conformer_sensitivity(
    sampled_cores_1k: list[str],
    mmps: pd.DataFrame,
    core_topo: dict[str, tuple[int, int]],
):
    print("\n" + "=" * 78)
    print("EXPERIMENT 5B: CONFORMER ENSEMBLE STABILITY")
    print("=" * 78)

    n_confs = 10
    print(f"Computing {n_confs} conformers for {len(sampled_cores_1k)} cores...")

    # Compute multi-conformer features for each core
    core_multiconf: dict[str, np.ndarray] = {}  # smi -> (n_confs, 9)
    t0 = time.perf_counter()
    for j, smi in enumerate(sampled_cores_1k):
        if (j + 1) % 200 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  {j+1}/{len(sampled_cores_1k)} ({elapsed:.0f}s)")
        feat_matrix = compute_3d_context_multi_conf(smi, n_confs=n_confs)
        core_multiconf[smi] = feat_matrix
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.0f}s")

    # ── CV per feature ──────────────────────────────────────────────────
    feature_names = [
        "n_donor", "n_acceptor", "n_hydrophobic", "n_aromatic",
        "sasa_attach", "gasteiger_charge", "n_rotbonds_2",
        "is_aromatic_attach", "n_heavy",
    ]

    cvs_per_feature = {fn: [] for fn in feature_names}
    for smi, feat_mat in core_multiconf.items():
        if feat_mat.shape[0] < 2:
            continue
        for fi, fn in enumerate(feature_names):
            col = feat_mat[:, fi]
            mean_val = np.mean(col)
            if mean_val > 1e-6:
                cv = float(np.std(col) / mean_val)
            else:
                cv = 0.0
            cvs_per_feature[fn].append(cv)

    print("\n--- Coefficient of Variation per feature (across conformers) ---")
    print(f"  {'Feature':<25s}  {'Mean CV':>8s}  {'Median CV':>10s}  {'Max CV':>8s}  {'Varies':>8s}")
    print("  " + "-" * 65)
    cv_summary = {}
    for fn in feature_names:
        vals = cvs_per_feature[fn]
        if not vals:
            continue
        mean_cv = float(np.mean(vals))
        median_cv = float(np.median(vals))
        max_cv = float(np.max(vals))
        frac_varies = float(np.mean([v > 0.01 for v in vals]))
        cv_summary[fn] = {
            "mean_cv": mean_cv, "median_cv": median_cv,
            "max_cv": max_cv, "frac_varies": frac_varies,
        }
        print(f"  {fn:<25s}  {mean_cv:>8.3f}  {median_cv:>10.3f}  "
              f"{max_cv:>8.3f}  {frac_varies:>7.1%}")

    # ── Ranking stability ───────────────────────────────────────────────
    print("\n--- Position ranking stability across conformers ---")

    # Build position data using conformer 0 features
    core_set = set(sampled_cores_1k)
    mmps_sub = mmps[mmps["core_smiles"].isin(core_set)]

    # Get molecules with >= 3 positions among sampled cores
    mol_positions = (
        mmps_sub[["mol_from", "core_smiles", "target_chembl_id"]]
        .drop_duplicates()
    )
    pos_count = mol_positions.groupby(["mol_from", "target_chembl_id"]).size().reset_index(name="n_pos")
    pos_count = pos_count[pos_count["n_pos"] >= 3]
    mol_positions = mol_positions.merge(
        pos_count[["mol_from", "target_chembl_id"]],
        on=["mol_from", "target_chembl_id"],
    )

    if len(mol_positions) == 0:
        print("  No molecules with >= 3 positions in sampled cores")
        return {"cv_summary": cv_summary, "ranking_stability": None}

    # For each conformer, predict sensitivity and rank
    # Use a global HGB trained on conformer 0 features for all data
    pos_agg = (
        mmps_sub
        .groupby(["core_smiles", "target_chembl_id"])
        .agg(sensitivity=("abs_delta_pActivity", "mean"), n_mmps=("abs_delta_pActivity", "count"))
        .reset_index()
    )
    pos_agg = pos_agg[pos_agg["n_mmps"] >= 3]
    mol_positions = mol_positions.merge(pos_agg, on=["core_smiles", "target_chembl_id"])
    mol_positions = mol_positions.sort_values(
        ["target_chembl_id", "mol_from", "core_smiles"]
    ).reset_index(drop=True)

    n_confs_actual = min(
        feat_mat.shape[0]
        for feat_mat in core_multiconf.values()
        if feat_mat.shape[0] > 0
    )
    n_confs_eval = min(n_confs, n_confs_actual)
    print(f"  Evaluating {n_confs_eval} conformers across "
          f"{mol_positions['mol_from'].nunique()} molecules")

    # Build feature matrices for each conformer
    rankings_per_conf = []  # list of arrays, each = predicted scores for all rows
    for ci in range(n_confs_eval):
        X_rows = []
        for _, row in mol_positions.iterrows():
            smi = row["core_smiles"]
            feat_mat = core_multiconf.get(smi)
            if feat_mat is not None and ci < feat_mat.shape[0]:
                ctx = feat_mat[ci]
            elif feat_mat is not None:
                ctx = feat_mat[0]  # fallback to first conformer
            else:
                ctx = np.zeros(9, dtype=np.float32)
            nh, nr = core_topo.get(smi, (0, 0))
            X_rows.append(np.concatenate([ctx, [float(nh), float(nr)]]))
        X_ci = np.array(X_rows, dtype=np.float32)

        # Use the simple heuristic for ranking (avoids training 10 models)
        # -core_n_heavy for the topology component, plus 3D features
        # Actually, just use the full feature vector with a single global model
        # Train on all data (no LOO needed — we just want ranking consistency)
        y_ci = mol_positions["sensitivity"].values.astype(np.float32)
        model = HistGradientBoostingRegressor(**HGB_KWARGS)
        model.fit(X_ci, y_ci)
        preds = model.predict(X_ci)
        rankings_per_conf.append(preds)

    # Compare rankings: for each (mol_from, target), check if top-1 changes
    n_mols = 0
    n_top1_changed = 0
    n_any_order_changed = 0

    for (mol_id, target), grp in mol_positions.groupby(["mol_from", "target_chembl_id"]):
        if len(grp) < 3:
            continue
        idx = grp.index.values
        n_mols += 1

        # Get rankings from each conformer
        top1s = set()
        orderings = []
        for ci in range(n_confs_eval):
            scores = rankings_per_conf[ci][idx]
            order = tuple(np.argsort(-scores))
            orderings.append(order)
            top1s.add(order[0])

        if len(top1s) > 1:
            n_top1_changed += 1
        if len(set(orderings)) > 1:
            n_any_order_changed += 1

    frac_top1 = n_top1_changed / n_mols if n_mols > 0 else 0.0
    frac_order = n_any_order_changed / n_mols if n_mols > 0 else 0.0

    print(f"\n  Molecules evaluated: {n_mols}")
    print(f"  Top-1 position changed across conformers: {n_top1_changed}/{n_mols} "
          f"({frac_top1:.1%})")
    print(f"  Any ordering changed: {n_any_order_changed}/{n_mols} "
          f"({frac_order:.1%})")

    ranking_result = {
        "n_mols": n_mols,
        "n_conformers": n_confs_eval,
        "frac_top1_changed": frac_top1,
        "frac_any_order_changed": frac_order,
    }

    return {"cv_summary": cv_summary, "ranking_stability": ranking_result}


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("OOD EXPERIMENT 5: FEATURE ROBUSTNESS / SENSITIVITY ANALYSIS")
    print("=" * 78)

    # Load MMP data first (needed for molecule-based sampling)
    mmps_path = Path("outputs/mmps/all_mmps.parquet")
    print(f"\nLoading MMP data from {mmps_path} ...")
    t0 = time.perf_counter()
    mmps_all = pd.read_parquet(
        mmps_path,
        columns=["target_chembl_id", "mol_from", "core_smiles", "abs_delta_pActivity"],
    )
    print(f"  Loaded {len(mmps_all):,} MMPs in {time.perf_counter()-t0:.0f}s")

    # Identify eligible molecules (>= 3 positions per mol-target)
    # so sampled cores form meaningful NDCG evaluation groups
    print("Identifying eligible molecules (>= 3 positions per mol-target) ...")
    pos_counts = (
        mmps_all
        .groupby(["core_smiles", "target_chembl_id"])
        .size()
        .reset_index(name="n_mmps")
    )
    pos_counts = pos_counts[pos_counts["n_mmps"] >= 3]

    mol_pos = (
        mmps_all[["mol_from", "core_smiles", "target_chembl_id"]]
        .drop_duplicates()
        .merge(pos_counts[["core_smiles", "target_chembl_id"]],
               on=["core_smiles", "target_chembl_id"])
    )
    cnt = (mol_pos.groupby(["mol_from", "target_chembl_id"])
           .size().reset_index(name="n_pos"))
    cnt = cnt[cnt["n_pos"] >= 3]
    mol_pos = mol_pos.merge(
        cnt[["mol_from", "target_chembl_id"]],
        on=["mol_from", "target_chembl_id"],
    )
    print(f"  {mol_pos['core_smiles'].nunique():,} eligible cores, "
          f"{len(cnt):,} eligible (mol, target) groups")

    # Sample molecules until we have ~2,000 unique cores
    rng = np.random.RandomState(42)
    unique_mol_ids = mol_pos["mol_from"].unique().copy()
    rng.shuffle(unique_mol_ids)

    sampled_cores_set: set[str] = set()
    sampled_mol_ids: set[int] = set()
    for mol_id in unique_mol_ids:
        mol_cores = set(mol_pos.loc[mol_pos["mol_from"] == mol_id, "core_smiles"])
        sampled_cores_set.update(mol_cores)
        sampled_mol_ids.add(int(mol_id))
        if len(sampled_cores_set) >= 2000:
            break

    sampled_cores = sorted(sampled_cores_set)
    print(f"  Sampled {len(sampled_mol_ids):,} molecules -> "
          f"{len(sampled_cores):,} unique cores")

    # Filter MMP data to sampled cores
    core_set = sampled_cores_set
    mmps = mmps_all[mmps_all["core_smiles"].isin(core_set)].reset_index(drop=True)
    del mmps_all, mol_pos, pos_counts, cnt
    print(f"  Filtered to {len(mmps):,} MMPs")

    # Compute core topology (radius-independent)
    print("Computing core topology features ...")
    core_topo = {}
    for smi in sampled_cores:
        core_topo[smi] = (_core_n_heavy(smi), _core_n_rings(smi))
    sampled_cores_1k = sampled_cores[:1000]

    # Run experiments
    results = {}
    results["exp5a"] = radius_sensitivity(sampled_cores, mmps, core_topo)
    results["exp5b"] = conformer_sensitivity(sampled_cores_1k, mmps, core_topo)

    # Save
    out_path = Path("outputs/ood/feature_sensitivity_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Make JSON-serializable
    def _clean(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    with open(out_path, "w") as f:
        json.dump(_clean(results), f, indent=2)
    print(f"\nResults saved to {out_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
