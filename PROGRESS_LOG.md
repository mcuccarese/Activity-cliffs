# Activity Cliffs — Adaptive Progress Log

> **How to use this file:** At the start of each work session, read the "Current Status" and "Next Steps" sections. Follow the exact prompts listed. After completing a step, note the date and any key results in the log. Claude Code will keep this file updated.

---

## Project at a Glance

| Layer | What exists | Status |
|---|---|---|
| Config | `src/activity_cliffs/config.py` | ✅ Written |
| Data loader | `src/activity_cliffs/data/chembl.py` | ✅ Written |
| Curation | `src/activity_cliffs/data/curation.py` | ✅ Written |
| Featurizer | `src/activity_cliffs/features/featurizer.py` | ✅ Written |
| Series assignment | `src/activity_cliffs/series/scaffold.py` | ✅ Written |
| Cliff miner | `src/activity_cliffs/cliffs/miner.py` | ✅ Written |
| Baseline models | `src/activity_cliffs/models/baselines.py` | ✅ Written |
| Contrastive model | `src/activity_cliffs/models/contrastive.py` | ✅ Written (MLP encoder, not full GNN) |
| Visualization | `src/activity_cliffs/analysis/visualization.py` | ✅ Written |
| End-to-end demo | `scripts/run_demo.py` | ✅ Written |
| Target browser | `scripts/list_targets.py` | ✅ Written |
| Demo notebook | `notebooks/activity_cliffs_demo.ipynb` | ✅ Written |
| **MMP extraction** | `src/activity_cliffs/data/mmp.py` | ✅ M2 — done |
| **MMP corpus (50 targets)** | `outputs/mmps/all_mmps.parquet` | ✅ M3 — done |
| **Real data run** | `outputs/demo/CHEMBL203/` | ✅ EGFR complete (M1) |
| **MMP feature matrix** | `outputs/features/mmp_features.parquet` | ✅ M4 — done |

---

## Milestone Map (revised 2026-03-17 — reframed around generalized Topliss / ShinkaEvolve)

| # | Milestone | Status | Est. sessions |
|---|---|---|---|
| M1 | Env setup + ChEMBL + EGFR end-to-end run | ✅ Done | — |
| M2 | MMP extraction module + validate on EGFR | ✅ Done | — |
| M3 | Scale MMP extraction to 50 targets, build transformation corpus | ✅ Done | — |
| M4 | Feature engineering: transformation features, context features, pre-cache | ✅ Done | — |
| M5 | ShinkaEvolve integration: template + fitness evaluator + first evolution | 🔲 | 1-2 |
| M6 | Analysis: read evolved programs, incorporate domain insight, iterate | 🔲 | 2-3 |
| M7 | Recommendation interface: "paste a molecule, get ranked modifications" | 🔲 | 1 |

---

## Completed Steps

| Date | Step | Result |
|---|---|---|
| 2026-03-16 | Installed Miniforge3 + initialized conda for PowerShell | conda works in PowerShell |
| 2026-03-16 | `conda env create -f environment.yml` + `pip install -e .` | `activity-cliffs` env active, package installed |
| 2026-03-16 | Smoke test | PyTorch + RDKit + activity_cliffs all import cleanly |
| 2026-03-17 | Downloaded ChEMBL 36 SQLite | `D:\Mike project data\...\chembl_36.db` |
| 2026-03-17 | Fixed `chembl.py` for ChEMBL 36 schema (`chembl_id` vs `target_chembl_id`) | SQL queries work |
| 2026-03-17 | Listed top 30 targets, selected **EGFR (CHEMBL203)** | 18,897 activities, 10,566 compounds |
| 2026-03-17 | EGFR end-to-end run (run_demo.py) | 1,693 pairs, 97 cliffs (5.7%), baseline ROC 0.63, contrastive ROC 0.74 |
| 2026-03-17 | Fixed RDKit MorganFP deprecation warnings (→ `rdFingerprintGenerator`) | Clean runs |
| 2026-03-17 | Full code review (Opus) | No blocking bugs. Reframed project toward generalized Topliss / ShinkaEvolve |
| 2026-03-17 | M2: MMP extraction module (mmp.py + extract_mmps.py) | 1,125,316 MMPs from EGFR; 6,613 unique cores; 334,830 cliff MMPs (29.8%); rdMMPA API fix needed (output format: empty core, fragments in t[1] as 'core.rgroup') |
| 2026-03-17 | M2b: Transformation enrichment analysis (Opus) | Meta-haloanilines are top potency boosters (I>Cl>F, reflecting hinge H-bond + hydrophobic pocket); NO2/tBu/NMe2 kill potency; simple descriptors explain <3% variance — context features critical for ShinkaEvolve |
| 2026-03-17 | M3: Scale MMP extraction to 50 Homo sapiens targets | **25,182,966 total MMPs**; 50 targets; cliff rates 11–35%; all saved to `outputs/mmps/`; combined at `outputs/mmps/all_mmps.parquet` |
| 2026-03-17 | M4: Feature engineering module (mmp_features.py + compute_mmp_features.py) | **298 MB** `outputs/features/mmp_features.parquet`; 25,182,966 rows × 11 columns; 4.2 min on single workstation; EGFR: 1,125,316 rows, 101 unique r=1 env hashes / 562 r=2; cliffs have larger delta_MW (96 vs 72) and delta_HAC (6.9 vs 5.1) — consistent with M2b finding that context features matter most |

---

## Current Status

**2026-03-17** — M4 complete. Feature matrix at `outputs/features/mmp_features.parquet` (298 MB, 25,182,966 rows, 11 feature columns). EGFR validation: 101 unique attachment environments at r=1, 562 at r=2; cliff pairs carry larger structural changes (delta_MW 96 vs 72, delta_HAC 6.9 vs 5.1) but the effect is weak, confirming that attachment context is the key discriminator. Ready for M5: ShinkaEvolve integration.

---

## Next Steps

### Step: M5 — ShinkaEvolve integration
**Model:** **Opus** for design (architecture decisions), then **Sonnet** for implementation
**Prompt to use (Opus first):**
```
Design the ShinkaEvolve integration for the activity cliff prediction system.
Context:
- Data: outputs/mmps/all_mmps.parquet (25M MMPs, 50 Homo sapiens targets)
- Features: outputs/features/mmp_features.parquet (11 cols: 7 delta-descriptors, 2 FP bytes, 2 env_hash uint32)
- Goal: evolve interpretable Python scoring functions that rank R-group swaps by predicted cliff probability
- Fitness metric: leave-one-target-out NDCG@k (predict which swaps are cliffs on held-out targets)
- The attachment environment hashes (env_hash_r1/r2) encode attachment context and are expected to be the most informative features
Design: (1) the ShinkaEvolve template for scoring functions, (2) the fitness evaluator, (3) the first evolution run setup.
```

---

## Model Decision Guide

| Task | Model | Why |
|---|---|---|
| Terminal errors / setup problems | Sonnet | Straightforward debugging |
| Running scripts, interpreting outputs | Sonnet | Clear-cut coding tasks |
| Reviewing ML results, tuning thresholds | Sonnet | Analytical but well-defined |
| Architectural decisions (e.g., should we add a GNN layer?) | Opus | High-stakes design choices |
| Debugging subtle training issues (loss not converging, leakage) | Opus | Complex reasoning needed |

---

## Key Parameters Reference

| Parameter | Default | What it controls |
|---|---|---|
| `--sim-min` | 0.85 | Tanimoto threshold for candidate pairs (lower = more pairs, more noise) |
| `--delta-min` | 1.5 | ΔpActivity to call a cliff (lower = more cliffs labeled) |
| `--min-series-size` | 15 (demo) / 10 (mine_cliffs) | Minimum scaffold series size |

Start with defaults. Only tune after seeing the first output metrics.

---

## Project Vision (revised 2026-03-17)

**Goal: A data-driven, generalized Topliss tree.** Given any starting compound and any target, recommend which structural modifications to test first to most efficiently discover steep SAR (activity cliffs).

**Architecture:**
1. **Data layer:** MMP extraction from ChEMBL (50-100 targets) → millions of (core, R-group swap, target, ΔpActivity) tuples
2. **Feature layer:** Transformation features (Δ-descriptors, attachment environment, R-group properties), context features (scaffold rigidity, pharmacophore proximity)
3. **Discovery layer:** ShinkaEvolve evolves interpretable Python scoring functions, evaluated by leave-one-target-out NDCG@k
4. **Application layer:** Paste a molecule → get ranked modification suggestions

**Why this matters:** Topliss encoded σ and π for phenyl substitution manually in 1972. We're learning the equivalent rules from all of medicinal chemistry, for any scaffold and any changeable motif. The evolved scoring functions ARE the scientific discovery — readable, interpretable, generalizable.

---

## M2 Design: MMP Extraction

**Module:** `src/activity_cliffs/data/mmp.py`
**Script:** `scripts/extract_mmps.py`
**Method:** `rdMMPA.FragmentMol` with `maxCuts=1` (single-cut, acyclic bonds only)
**Algorithm:** Fragment all molecules → group by canonical core → pair within groups → record transformation
**Output:** Parquet with core_smiles, rgroup_from, rgroup_to, transform_smarts, delta_pActivity
**Scale control:** Cap group size at 200 molecules per core

---

## Known Gaps

1. **Model persistence** — Contrastive encoder doesn't save weights. Low priority given project reframe.
2. **Atom-level attribution** — Not yet implemented. May become relevant in M6 analysis.
3. **No GNN** — Fine for now. Graph-level features may be useful in M4 featurization.
