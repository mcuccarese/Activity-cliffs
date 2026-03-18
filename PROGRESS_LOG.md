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
| **ShinkaEvolve integration** | `evolve/` (initial.py, evaluate.py, run_evo.py) | ✅ M5 — code + manual evolution |
| **Eval data** | `evolve/eval_data/eval_data.npz` | ✅ 1.19M rows, 50 targets, 12 features, 14.8 MB |
| **Manual evolution harness** | `evolve/manual_evolve.py` | ✅ Claude-as-LLM evolution loop |
| **Candidate functions** | `evolve/candidates/gen1_*.py`, `gen2_*.py` | ✅ 16 candidates tested |

---

## Milestone Map (revised 2026-03-18 — 3D context × change-type architecture)

| # | Milestone | Status | Model |
|---|---|---|---|
| M1 | Env setup + ChEMBL + EGFR end-to-end run | ✅ Done | — |
| M2 | MMP extraction module + validate on EGFR | ✅ Done | — |
| M3 | Scale MMP extraction to 50 targets | ✅ Done | — |
| M4 | Feature engineering (2D) | ✅ Done | — |
| M5 | ShinkaEvolve + manual evolution + ML ceiling analysis | ✅ Done | — |
| M6a | **3D context features at attachment points** | 🔲 NEXT | Sonnet |
| M6b | **Change-type classification of R-groups** | 🔲 | Sonnet |
| M6c | **Interaction feature test (break 0.52 ceiling?)** | 🔲 | Sonnet + Opus |
| M7 | **Webapp: SAR Sensitivity Explorer** | 🔲 | Sonnet |

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
| 2026-03-17 | M5: ShinkaEvolve integration (Opus design + implementation) | `evolve/initial.py` (template scoring fn), `evolve/evaluate.py` (NDCG@5 fitness), `evolve/run_evo.py` (launcher); `scripts/prepare_evolve_data.py` subsampled 1,186,402 rows (200 mol/target × 50 targets) into 14.8 MB eval_data.npz; baseline NDCG@5 = 0.513 (17% above random 0.439); eval runs in 0.03s; `pip install shinka-evolve` done |
| 2026-03-17 | M5b: Manual evolution — Claude Code as LLM mutation operator | Built `evolve/manual_evolve.py` harness (load any .py, eval NDCG@5 instantly). Rebuilt eval_data with **12 features** (added 4 env_hash context features: env_r1/r2_cliff_rate, env_r1/r2_mean_delta from pooled ChEMBL statistics). Ran 2 generations (16 candidates): Gen1 tested 9 hypotheses on 8 features (L2 norm, LogP-dominant, interactions, signed, thresholds, dissimilarity-focus, rank-based, max-feature); Gen2 tested env context features (env-only, multiplicative, gated-threshold, kitchen-sink). **Key finding: env context features don't help NDCG because they have low within-group variance** (NDCG ranks within a mol_from group; all transforms from the same molecule share similar attachment environments). Best NDCG@5 = 0.5136 (L2 norm of normalised deltas). Next: measure ML ceiling with HistGradientBoosting to determine if feature set is fundamentally limited or if we need a better scoring function architecture. |
| 2026-03-17 | M5c: ML ceiling measurement (v1) — HistGradientBoosting on original features | HGB 8 feat LOO-target = **0.5178** (barely beats L2 norm 0.5136). HGB augmented 17 feat = 0.5127. HGB 12 feat (+ env) = 0.5165. HGB deep (500 iter) = 0.5171. **Conclusion: feature set fundamentally limited — nonlinear ML gains < 0.005 over L2 norm.** |
| 2026-03-17 | M5c: Anti-gaming/overfitting plan | Wrote `evolve/ANTI_GAMING_PLAN.md`: identified 8 risks (eval data memorization, NDCG gaming, "bigger is better" degeneracy, feature correlation exploitation, complexity creep, within-group collapse, symmetric MMP leakage, target-class confounding). Defined 3-phase mitigation plan + 6 success criteria for generalizable scoring functions. |
| 2026-03-17 | M5c: Enriched feature pipeline (v2) | Built `scripts/prepare_evolve_data_v2.py` adding 35 new features: FP XOR PCA (20d, 20% variance), 12 FG net-change flags (halogen, amine, hydroxyl, carboxyl, amide, sulfonamide, nitro, nitrile, aromatic, methyl, ether, carbonyl), log transform frequency, R-group size ratio, max size. Saved as `evolve/eval_data/eval_data_v2.npz` (151.6 MB, 1.19M rows x 43 features). |
| 2026-03-17 | M5c: ML ceiling measurement (v2) — enriched features | **Enriched features DO NOT break the ceiling.** FG flags only = 0.4935 (worse). XOR PCA only = 0.5058 (worse). Abs deltas + FG = 0.5144 (tied). All 43 features = 0.5151 (tied). Best mix = 0.5137 (tied). **Original 8 abs deltas + dissim = 0.5170 remains best.** The ~0.52 ceiling is a fundamental limit of target-agnostic 2D descriptors. |
| 2026-03-18 | M6a: Architecture decision — 3D context × change-type hybrid | Reviewed all 3D featurization methods (classical through Uni-Mol/SchNet). Chose Hybrid Approach C: `interpretable = 3D_context @ W @ change_type` + `learned_residual = MLP(atom_embed, rgroup_embed)`. Key insight: context × change_type interaction provides within-group variance (the missing piece). Webapp vision: input SMILES → positions colored by SAR sensitivity → ranked change-type recommendations. Session died to 529 errors before updating files — recovered in next session. |

---

## Current Status

**2026-03-18** — M6a architecture decision complete (discussed in session, lost to context limit). Chose **Hybrid Approach C: 3D context × change-type interaction + learned residual**.

**Key architectural insight:** The 2D ceiling (~0.52) exists because env_hash is constant within a mol_from group — NDCG measures within-group ranking, so features that don't vary within the group contribute zero signal. The **context × change_type cross-product** solves this: even when all transforms share the same attachment point (same 3D context), different R-group types (EWG vs lipophilic vs polar) interact differently with that context. The within-group variance comes from the **interaction term**, not from either factor alone. This was the missing piece.

**Webapp vision:** Input any compound → fragment at all positions → color positions by SAR sensitivity → click a position to see ranked change-type recommendations ("Try EWG (0.72) > lipophilic (0.65)"). Predicts cliff probability, not direction.

---

## Next Steps

### Step: M6a — Compute 3D pharmacophore context features
**Model:** Sonnet
**Prompt:**
```
Read CONTINUATION_PLAN.md and PROGRESS_LOG.md. We've decided on a 3D context ×
change-type hybrid architecture to break the 0.52 NDCG@5 ceiling (full details
in the continuation plan).

Build a script to compute 3D pharmacophore context features at each MMP
attachment point. For each unique core_smiles in our MMP dataset:

1. Parse the core SMILES (contains [*:1] attachment point)
2. Generate a 3D conformer with ETKDG (AllChem.EmbedMolecule + MMFF optimize)
3. At the attachment atom ([*:1] or its neighbor), compute:
   a. Pharmacophore environment: count donor/acceptor/hydrophobic/aromatic
      features within 4Å of the attachment atom
   b. Steric accessibility: SASA or Labute ASA contribution at/near cut atom
   c. Electrostatic character: Gasteiger partial charge at the neighbor atom
   d. Local rigidity: rotatable bonds within 2 bonds of attachment
   e. Aromatic context: is the attachment on an aromatic ring?

4. Save as a lookup table (core_smiles → feature vector) in parquet or npz

Start with a small test (100 cores from EGFR) to validate the approach,
then scale to all ~104K unique cores. The features should be 5-10 interpretable
floats per attachment point.

Use the conda env python: c:/Users/mcucc/miniforge3/envs/activity-cliffs/python
ChEMBL DB: "D:\Mike project data\Activity cliffs\chembl_36\chembl_36_sqlite\chembl_36.db"
```

### Step: M6b — Classify R-group transforms into change types
**Model:** Sonnet
**Prompt:**
```
Read CONTINUATION_PLAN.md. Build a module to classify each R-group fragment
into medchem-meaningful change types using SMARTS patterns:
- EWG (F, Cl, Br, CF3, NO2, CN, SO2, COOH)
- EDG (NH2, OH, OMe, NMe2, alkyl donors)
- Lipophilic (alkyl chains, aromatic rings, halogenated alkyl)
- Polar (OH, NH, CONH, SO2NH)
- H-bond donor gain/loss
- H-bond acceptor gain/loss
- Size increase / decrease
- Ring gain / loss
- Aromaticity change

Each transform (rgroup_from → rgroup_to) gets a change-type vector.
Test on EGFR MMPs, then apply to all 25M rows.
```

### Step: M6c — Test interaction features against the ceiling
**Model:** Sonnet (implementation) + Opus (interpretation)
**Prompt:** Build context × change_type cross-product features. Train HGB.
Test whether NDCG@5 > 0.52. If yes → proceed to hybrid model + webapp.

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

## Project Vision (revised 2026-03-18 — 3D context × change-type hybrid)

**Goal: SAR Sensitivity Explorer.** Given any compound, highlight which positions are most SAR-sensitive and recommend what TYPE of structural change to try first (EWG, lipophilic, polar, etc.). Predicts cliff probability, not direction — helps the chemist narrow scope for the first round of modifications.

**Architecture (Hybrid Approach C):**
1. **Data layer:** 25M MMPs from 50 ChEMBL targets ✅
2. **Feature layer:** 2D Δ-descriptors ✅ + NEW: 3D pharmacophore context at attachment + change-type classification of R-groups + context × change_type cross-products
3. **Scoring layer:** `interpretable_score = 3D_context @ W @ change_type` (Topliss rules) + `learned_residual = MLP(atom_embedding, rgroup_embedding)` → `final = interpretable + α * residual`
4. **Webapp:** Streamlit/Gradio molecule viewer → positions colored by sensitivity → click for ranked change-type recommendations

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

1. **2D ceiling (~0.52) understood — solution designed but untested.** The context × change_type interaction is hypothesized to break the ceiling by providing within-group variance. Needs M6a-c to validate.
2. **3D context features not yet computed** — Need ETKDG conformers + pharmacophore env for ~104K unique cores. This is M6a.
3. **Change-type categories not yet defined** — R-group SMARTS classification into EWG/EDG/lipophilic/polar/etc. This is M6b.
4. **Interaction features not yet tested** — The context × change_type cross-products need to be evaluated against the ceiling. This is M6c.
5. **ShinkaEvolve automated evolution untested** — Manual evolution via Claude Code works as alternative.
6. **Anti-gaming guardrails designed but not implemented** — `evolve/ANTI_GAMING_PLAN.md` ready.
