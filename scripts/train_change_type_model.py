#!/usr/bin/env python
"""
M9: Train the change-type cliff model.

For each fragmentable position (defined by its 9D pharmacophore context),
predicts which R-group property dimension causes the largest activity swings
(|Δ pActivity|) when changed.  This is a Topliss-style "start here first"
recommendation — no outcome direction is asserted.

Model:
    Input  (20D): [9D pharmacophore context | 11D Δ R-group property vector]
    Target (1D) : abs_delta_pActivity

At inference, the model is probed along each of the 11 Δ-prop axes
independently at ±1σ (data-derived), and the axis with the highest predicted
|Δ| is ranked first.

Usage:
    python scripts/train_change_type_model.py
    python scripts/train_change_type_model.py --n-rows 10000000
    python scripts/train_change_type_model.py --skip-validation
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from activity_cliffs.features.context_3d import CONTEXT_3D_FEATURES
from activity_cliffs.features.change_type import RGROUP_PROP_NAMES, CHANGE_TYPE_NAMES

MMPS_PATH        = ROOT / "outputs/mmps/all_mmps.parquet"
CONTEXT_3D_PATH  = ROOT / "outputs/features/context_3d.parquet"
RGROUP_PROPS_PATH = ROOT / "outputs/features/rgroup_props.parquet"
MODEL_DIR        = ROOT / "webapp/model"

# ── Human-readable labels per Δ-prop axis ────────────────────────────────────
AXIS_LABELS: dict[str, str] = {
    "delta_has_ewg":       "EWG character change",
    "delta_has_edg":       "EDG character change",
    "delta_ewg_count":     "EWG count change",
    "delta_edg_count":     "EDG count change",
    "delta_n_hbd":         "H-bond donor change",
    "delta_n_hba":         "H-bond acceptor change",
    "delta_lipophilicity": "Lipophilicity change",
    "delta_heavy_atoms":   "Size change",
    "delta_n_rings":       "Ring count change",
    "delta_n_arom_rings":  "Aromatic ring change",
    "delta_fsp3":          "Saturation change",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
sys.stdout.reconfigure(line_buffering=True)

app = typer.Typer(add_completion=False)


@app.command()
def main(
    n_rows: int = typer.Option(
        5_000_000,
        "--n-rows",
        help="Number of MMP rows to sample for training (stratified by target).",
    ),
    skip_validation: bool = typer.Option(
        False,
        "--skip-validation",
        help="Skip LOO-target Spearman validation (faster for iteration).",
    ),
    seed: int = typer.Option(42, help="Random seed."),
) -> None:
    """Train the M9 change-type cliff model and save to webapp/model/."""
    t_start = time.perf_counter()
    rng = np.random.RandomState(seed)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load MMP table (only needed columns) ──────────────────────────
    logger.info("Loading MMP table from %s ...", MMPS_PATH)
    mmps = pd.read_parquet(
        MMPS_PATH,
        columns=["target_chembl_id", "core_smiles", "rgroup_from", "rgroup_to",
                 "abs_delta_pActivity"],
    )
    n_total = len(mmps)
    n_targets = mmps["target_chembl_id"].nunique()
    logger.info("  %d rows, %d targets", n_total, n_targets)

    # ── Step 2: Stratified sample ─────────────────────────────────────────────
    n_sample = min(n_rows, n_total)
    if n_sample < n_total:
        logger.info("Stratified sampling %d rows across %d targets ...", n_sample, n_targets)
        target_counts = mmps["target_chembl_id"].value_counts()
        fracs = (target_counts / n_total * n_sample).round().astype(int)
        fracs = fracs.clip(lower=1)
        # Adjust to hit exactly n_sample
        diff = n_sample - fracs.sum()
        if diff != 0:
            fracs.iloc[0] += diff

        parts: list[pd.DataFrame] = []
        for target, count in fracs.items():
            group = mmps[mmps["target_chembl_id"] == target]
            take = min(count, len(group))
            parts.append(group.sample(n=take, random_state=rng.randint(0, 2**31)))
        mmps = pd.concat(parts, ignore_index=True)
        logger.info("  Sampled %d rows", len(mmps))
    else:
        logger.info("Using full dataset (%d rows)", n_total)

    # ── Step 3: Build context lookup (core_smiles → 9D float32) ──────────────
    logger.info("Loading 3D context from %s ...", CONTEXT_3D_PATH)
    ctx_df = pd.read_parquet(CONTEXT_3D_PATH)
    ctx_lookup: dict[str, np.ndarray] = dict(
        zip(ctx_df["core_smiles"], ctx_df[CONTEXT_3D_FEATURES].values.astype(np.float32))
    )
    logger.info("  %d cores in lookup", len(ctx_lookup))

    # ── Step 4: Build R-group property lookup (smi → 11D float32) ────────────
    logger.info("Loading R-group properties from %s ...", RGROUP_PROPS_PATH)
    rg_df = pd.read_parquet(RGROUP_PROPS_PATH)
    rg_lookup: dict[str, np.ndarray] = dict(
        zip(rg_df["rgroup_smiles"], rg_df[RGROUP_PROP_NAMES].values.astype(np.float32))
    )
    logger.info("  %d R-groups in lookup", len(rg_lookup))

    # ── Step 5: Build feature matrix ─────────────────────────────────────────
    logger.info("Building feature matrix (20D) ...")
    n = len(mmps)
    n_ctx = len(CONTEXT_3D_FEATURES)   # 9
    n_prop = len(RGROUP_PROP_NAMES)    # 11

    zero_ctx  = np.zeros(n_ctx,  dtype=np.float32)
    zero_prop = np.zeros(n_prop, dtype=np.float32)

    X_ctx  = np.empty((n, n_ctx),  dtype=np.float32)
    X_delta = np.empty((n, n_prop), dtype=np.float32)
    n_miss_ctx = 0
    n_miss_rg  = 0

    for i, (core, rg_from, rg_to) in enumerate(
        zip(mmps["core_smiles"], mmps["rgroup_from"], mmps["rgroup_to"])
    ):
        ctx = ctx_lookup.get(core)
        if ctx is None:
            X_ctx[i] = zero_ctx
            n_miss_ctx += 1
        else:
            X_ctx[i] = ctx

        pf = rg_lookup.get(rg_from)
        pt = rg_lookup.get(rg_to)
        if pf is None:
            pf = zero_prop
            n_miss_rg += 1
        if pt is None:
            pt = zero_prop
            n_miss_rg += 1
        X_delta[i] = pt - pf

        if i > 0 and i % 1_000_000 == 0:
            logger.info("  Built %d / %d rows ...", i, n)

    X = np.column_stack([X_ctx, X_delta])   # (n, 20)
    y = mmps["abs_delta_pActivity"].values.astype(np.float32)
    logger.info("  Feature matrix: %s  (miss ctx=%d, miss rg=%d)", X.shape, n_miss_ctx, n_miss_rg)

    # Compute per-axis σ from training Δ-prop values (used at inference)
    delta_prop_sigmas: dict[str, float] = {}
    for j, name in enumerate(CHANGE_TYPE_NAMES):
        sigma = float(np.std(X_delta[:, j]))
        delta_prop_sigmas[name] = round(sigma, 4)
    logger.info("  Per-axis Δ-prop σ: %s", delta_prop_sigmas)

    # ── Step 6: LOO-target Spearman validation ────────────────────────────────
    loo_spearmans: list[float] = []

    if skip_validation:
        logger.info("Skipping LOO-target validation (--skip-validation).")
    else:
        logger.info("LOO-target Spearman r validation (%d targets) ...", n_targets)
        target_ids = mmps["target_chembl_id"].values
        unique_targets = np.unique(target_ids)

        for i_t, held_out in enumerate(unique_targets):
            mask_test  = target_ids == held_out
            mask_train = ~mask_test

            if mask_test.sum() < 20:
                continue  # too few test rows to be meaningful

            X_tr, y_tr = X[mask_train], y[mask_train]
            X_te, y_te = X[mask_test],  y[mask_test]

            # Cap test set to avoid very long eval on large targets
            if len(X_te) > 50_000:
                idx = rng.choice(len(X_te), size=50_000, replace=False)
                X_te, y_te = X_te[idx], y_te[idx]

            loo_model = HistGradientBoostingRegressor(
                max_iter=300, max_depth=6, learning_rate=0.1,
                min_samples_leaf=100, random_state=seed,
            )
            loo_model.fit(X_tr, y_tr)
            y_hat = loo_model.predict(X_te)
            r = float(spearmanr(y_te, y_hat).statistic)
            loo_spearmans.append(r)
            logger.info(
                "  [%2d/%2d] %-20s  test=%5d  Spearman r=%.3f",
                i_t + 1, len(unique_targets), held_out, len(y_te), r,
            )

        logger.info(
            "LOO Spearman: mean=%.3f  std=%.3f  min=%.3f",
            np.mean(loo_spearmans), np.std(loo_spearmans), np.min(loo_spearmans),
        )

    # ── Step 7: Train final model on all sampled data ─────────────────────────
    logger.info("Training final HGB on %d rows ...", len(X))
    t0 = time.perf_counter()
    model = HistGradientBoostingRegressor(
        max_iter=300,
        max_depth=6,
        learning_rate=0.1,
        min_samples_leaf=100,
        random_state=seed,
    )
    model.fit(X, y)
    logger.info("  Trained in %.1fs", time.perf_counter() - t0)

    # ── Step 8: Sanity check — expected cliff ranking for archetypal contexts ──
    logger.info("Sanity check: probing canonical pharmacophore contexts ...")

    def _probe_context(label: str, ctx: dict[str, float]) -> None:
        ctx_vec = np.array([ctx.get(f, 0.0) for f in CONTEXT_3D_FEATURES], dtype=np.float32)
        scores: list[tuple[float, str]] = []
        for j, ax_name in enumerate(CHANGE_TYPE_NAMES):
            sigma = delta_prop_sigmas[ax_name]
            if sigma < 1e-6:
                sigma = 1.0
            delta_pos = np.zeros(n_prop, dtype=np.float32); delta_pos[j] = sigma
            delta_neg = np.zeros(n_prop, dtype=np.float32); delta_neg[j] = -sigma
            x_pos = np.concatenate([ctx_vec, delta_pos]).reshape(1, -1)
            x_neg = np.concatenate([ctx_vec, delta_neg]).reshape(1, -1)
            pred = max(float(model.predict(x_pos)[0]), float(model.predict(x_neg)[0]))
            scores.append((pred, AXIS_LABELS[ax_name]))
        scores.sort(reverse=True)
        top3 = ", ".join(f"{lbl} ({sc:.2f})" for sc, lbl in scores[:3])
        logger.info("  [%s] Top 3: %s", label, top3)

    # Hydrophobic pocket: high hydrophobic + high SASA, no donors
    _probe_context("Hydrophobic pocket", {
        "n_hydrophobic_4A": 8.0, "sasa_attach": 40.0,
        "n_donor_4A": 0.0, "n_acceptor_4A": 1.0, "n_aromatic_4A": 4.0,
        "gasteiger_charge": -0.05, "n_rotbonds_2": 1.0,
        "is_aromatic_attach": 1.0, "n_heavy_4A": 8.0,
    })
    # H-bond donor pocket: high donors, low hydrophobic
    _probe_context("Donor-rich pocket", {
        "n_donor_4A": 4.0, "n_acceptor_4A": 2.0, "n_hydrophobic_4A": 1.0,
        "n_aromatic_4A": 0.0, "sasa_attach": 20.0, "gasteiger_charge": -0.15,
        "n_rotbonds_2": 0.0, "is_aromatic_attach": 0.0, "n_heavy_4A": 5.0,
    })
    # Sterically crowded: many heavy atoms nearby, low SASA
    _probe_context("Crowded site", {
        "n_heavy_4A": 15.0, "sasa_attach": 5.0, "n_hydrophobic_4A": 5.0,
        "n_donor_4A": 1.0, "n_acceptor_4A": 1.0, "n_aromatic_4A": 6.0,
        "gasteiger_charge": 0.0, "n_rotbonds_2": 0.0, "is_aromatic_attach": 1.0,
    })

    # ── Step 9: Save model and metadata ───────────────────────────────────────
    model_path = MODEL_DIR / "change_type_hgb.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Saved model → %s (%.0f KB)", model_path, model_path.stat().st_size / 1024)

    feature_names = [f"ctx_{c}" for c in CONTEXT_3D_FEATURES] + CHANGE_TYPE_NAMES
    meta: dict = {
        "feature_names": feature_names,
        "context_feature_names": [f"ctx_{c}" for c in CONTEXT_3D_FEATURES],
        "delta_prop_names": CHANGE_TYPE_NAMES,
        "axis_labels": AXIS_LABELS,
        "n_training_rows": int(len(X)),
        "n_targets_trained_on": int(n_targets),
        "hyperparameters": {
            "max_iter": 300, "max_depth": 6,
            "learning_rate": 0.1, "min_samples_leaf": 100,
        },
        "y_mean": float(y.mean()),
        "y_std": float(y.std()),
        "delta_prop_sigmas": delta_prop_sigmas,
    }
    if loo_spearmans:
        meta["loo_spearman_mean"] = float(np.mean(loo_spearmans))
        meta["loo_spearman_std"]  = float(np.std(loo_spearmans))
        meta["loo_spearman_min"]  = float(np.min(loo_spearmans))
        meta["loo_spearman_per_target"] = {
            str(t): float(r)
            for t, r in zip(unique_targets[: len(loo_spearmans)], loo_spearmans)
        }

    meta_path = MODEL_DIR / "change_type_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Saved metadata → %s", meta_path)

    logger.info(
        "Done in %.1fs total.  Run: streamlit run webapp/app.py",
        time.perf_counter() - t_start,
    )


if __name__ == "__main__":
    app()
