# Activity Cliffs → Generalized Topliss Tree: Continuation Plan

> **Purpose:** Complete context for picking up this project in a new Claude Code session. Read this file first, then PROGRESS_LOG.md for step-by-step history.

---

## 1. What This Project Is

A research prototype that mines medicinal chemistry SAR data to build a **data-driven, generalized Topliss decision tree**: given any starting compound, recommend which structural modifications to try first to most efficiently discover activity cliffs (steep SAR).

The original 1972 Topliss tree encoded σ (electronic) and π (hydrophobic) Hansch parameters for phenyl substitution as a hand-crafted decision tree. We're learning the equivalent rules from the entire history of medicinal chemistry in ChEMBL, for any scaffold and any changeable motif, using **ShinkaEvolve** (Sakana AI's LLM-driven evolutionary program optimization) to discover interpretable scoring functions.

---

## 2. What's Done (as of 2026-03-18)

### Environment
- **OS:** Windows 11 Pro, RTX 4070
- **Python:** Miniforge3 → conda env `activity-cliffs` (Python 3.11, PyTorch, RDKit)
- **Package:** installed editable via `pip install -e .`
- **ChEMBL 36 SQLite:** `D:\Mike project data\Activity cliffs\chembl_36\chembl_36_sqlite\chembl_36.db`
- **Env var:** `CHEMBL_SQLITE_PATH` set via `setx` (requires terminal restart to pick up; alternatively pass `--chembl-sqlite "D:\Mike project data\Activity cliffs\chembl_36\chembl_36_sqlite\chembl_36.db"` directly)

### Code fixes applied
- `chembl.py`: Changed `td.target_chembl_id` → `td.chembl_id` in all SQL queries (ChEMBL 36 schema change)
- `featurizer.py`, `baselines.py`, `contrastive.py`: Replaced deprecated `AllChem.GetMorganFingerprintAsBitVect` with `rdFingerprintGenerator.GetMorganGenerator`

### M1 Results: EGFR (CHEMBL203) end-to-end run
- **Curated molecules:** 10,566
- **Scaffold series:** 3,864
- **Pairs mined:** 1,693 (Tanimoto ≥ 0.85, series size ≥ 15)
- **Cliff pairs:** 97 (5.7% cliff rate)
- **Baseline ROC-AUC:** 0.630 (LogReg on XOR fingerprints)
- **Contrastive ROC-AUC:** 0.736 (MLP encoder, 8 epochs)
- **Outputs:** `outputs/demo/CHEMBL203/` — parquet files, metrics JSON, plots

### Code review findings (Opus, 2026-03-17)
- No blocking bugs in any module
- Code quality 7.5-8/10
- Correct scaffold-level splitting (GroupShuffleSplit on series_id) prevents leakage
- SALI-like cliff score, standard contrastive loss, MCS-based pair visualization

### M5c Results: ML Ceiling Analysis (2026-03-17)
- **2D feature ceiling confirmed at NDCG@5 ~ 0.52** across all feature sets and models tested
- HistGradientBoosting (8 feat, LOO-target): 0.5178 — barely beats L2 norm (0.5136)
- Enriched v2 features (43 dim: FP XOR PCA, FG flags, transform freq, size): no improvement
- FG net flags alone: 0.4935 (worse than property deltas alone)
- Anti-gaming plan written: 8 risks identified, 3-phase mitigation, 6 success criteria

### M6a Decision: 3D Context × Change-Type Architecture (2026-03-18)
- **Reviewed all 3D featurization methods** — classical (pharmacophore, steric, electrostatic) through pretrained ML (Uni-Mol, SchNet, DimeNet, PaiNN, 3D-Infomax)
- **Key insight:** Global 3D shape descriptors (PMI, USR, WHIM) don't help — the question is per-position, not whole-molecule. Local features (pharmacophore env at attachment, steric accessibility, electrostatic potential) are directly relevant.
- **ML 3D models** (Uni-Mol, SchNet etc.) produce powerful per-atom embeddings but don't natively decompose into "what does this mean for halogen vs amine addition" — they capture WHERE but not WHERE × WHAT.
- **Chose Hybrid Approach (Approach C):** `interpretable_score = classical_3D_context @ W @ change_type` + `learned_residual = MLP(atom_embedding, rgroup_embedding)` with tunable alpha. The classical part gives the Topliss-style "swap this with an EWG" recommendation; the learned part captures subtleties.
- **This solves the within-group variance problem:** Previous env_hash was constant within a mol_from group. The context × change_type interaction varies because different R-groups interact differently with the same 3D context — the within-group signal comes from the cross-term.
- **Webapp vision confirmed:** Input SMILES → 3D conformer → fragment at all single-cut positions → score all change types per position → heatmap of SAR sensitivity + ranked change-type recommendations per position

---

## 3. The Big Idea: Why MMPs, Not Cliff Pairs

The user (medicinal chemistry domain expert, not a coder) clarified the vision: the goal is NOT to predict whether a pair is a cliff, but to predict **which structural modification to try first** to most efficiently explore SAR — a generalized Topliss tree.

**Why Matched Molecular Pairs (MMPs) are the right data unit:**
- A cliff pair (Tanimoto ≥ 0.85) can differ at 3-4 positions — you can't attribute the activity change to any single modification
- An MMP differs by exactly one R-group swap at one position — the ΔpActivity is unambiguously caused by that one change
- MMPs capture ALL transformations (informative and uninformative), not just high-similarity cliffs
- Every MMP in ChEMBL is a recorded experiment: "someone swapped R-group X for Y on scaffold Z at target T, and potency changed by Δ"

**The question changes from:**
- "Is this pair a cliff?" (classification, ROC-AUC)
- TO: "Which modification is most likely to reveal steep SAR?" (ranking, NDCG@k)

---

## 4. Architecture (revised 2026-03-18 — 3D Context × Change-Type Hybrid)

```
┌─────────────────────────────────────────────────────────────┐
│  DATA LAYER (M2-M3) ✅                                      │
│  ChEMBL → curate → MMP extraction → transformation dataset │
│  50 targets, 25M MMPs (core, R_from, R_to, target, Δp)     │
├─────────────────────────────────────────────────────────────┤
│  FEATURE LAYER (M4 + M6a-NEW)                               │
│  2D features (done): Δ-descriptors, FG flags, env hash     │
│  3D context features (new): pharmacophore env at attachment, │
│    steric accessibility, electrostatic potential at cut atom │
│  Change-type categories (new): EWG, EDG, lipophilic, polar, │
│    H-bond donor/acceptor, size↑/↓, ring gain/loss (~8-10)   │
│  Interaction features: 3D_context × change_type cross-terms │
├─────────────────────────────────────────────────────────────┤
│  SCORING LAYER (Hybrid Approach C)                          │
│  interpretable = classical_3D_context @ W @ change_type     │
│  learned_residual = MLP(atom_embedding, rgroup_embedding)   │
│  final_score = interpretable + α * learned_residual         │
│  Fitness: leave-one-target-out NDCG@k                       │
├─────────────────────────────────────────────────────────────┤
│  APPLICATION LAYER (M7 — Webapp)                            │
│  Input: SMILES of compound or series                        │
│  → 3D conformer → fragment at all single-cut positions      │
│  → per-position: score all change types                     │
│  Output: molecule viewer with SAR sensitivity heatmap       │
│    + ranked change-type recommendations per position        │
│    "Position 4: Try EWG (0.72) > lipophilic (0.65)"         │
└─────────────────────────────────────────────────────────────┘
```

### Why the Hybrid Solves the Within-Group Problem

Previous attempts failed because env_hash was constant within a mol_from group — NDCG ranks within a group, so features that don't vary within the group contribute zero signal. The **context × change_type interaction** fixes this: even when all transforms share the same attachment point (same 3D context), the interaction term varies because different R-groups interact differently with the same context. "Hydrophobic pocket × halogen" scores differently from "hydrophobic pocket × amine." The within-group variance comes from the **cross-term**, not from either factor alone.

### 3D Feature Tiers (reviewed 2026-03-18)

| Method | Local/Global | R-group relevance |
|---|---|---|
| **Pharmacophore env at attachment** (donor/acceptor/hydrophobic/aromatic) | Local | Direct — "hydrophobic region in R-group direction" |
| **Steric accessibility** (SASA around cut atom) | Local | Direct — crowded positions penalize bulky R-groups |
| **Electrostatic potential** (Gasteiger charges at cut atom) | Local | Maps to EWG/EDG sensitivity |
| PMI / Asphericity / USR | Global | Weak — doesn't distinguish positions |
| 3D autocorrelation / WHIM / GETAWAY | Global | Weak — whole-molecule QSAR |
| **Uni-Mol / SchNet / PaiNN** (per-atom embeddings) | Local | Powerful but black-box — use for learned residual |
| **3D-Infomax** (contrastive 2D/3D) | Local | Bonus: can use 2D input at inference |

---

## 5. Milestone Plan

### M2–M5 — COMPLETED (see Progress Log for details)

- M2: MMP extraction (1.1M MMPs from EGFR; rdMMPA single-cut)
- M2b: Transformation enrichment analysis (halogens at hinge binders → top cliff drivers)
- M3: Scaled to 50 targets (25M total MMPs)
- M4: Feature engineering (25M rows × 11 columns, 298 MB parquet)
- M5: ShinkaEvolve integration + manual evolution (16 candidates, best NDCG@5 = 0.5178)
- M5c: ML ceiling confirmed at ~0.52 with 2D features

### M6a — 3D Context Features (NEXT)

**Goal:** Compute local 3D pharmacophore context features at each attachment point for the ~104K unique cores in the dataset.

**What to compute per attachment atom:**
1. **3D conformer generation** — RDKit ETKDG (`AllChem.EmbedMolecule` + `AllChem.MMFFOptimizeMolecule`)
2. **Pharmacophore environment** — count donor/acceptor/hydrophobic/aromatic/positive/negative pharmacophore features within 4Å of the attachment atom (RDKit `Chem.Pharm2D` or manual check of atom types)
3. **Steric accessibility** — solvent-accessible surface area contribution at the cut atom (`rdFreeSASA.CalcSASA` or `Descriptors.LabuteASA`)
4. **Electrostatic character** — Gasteiger partial charge at the cut atom
5. **Local rigidity** — number of rotatable bonds within 2 bonds of the attachment

**Output:** ~5-10 interpretable features per attachment point, cached as a lookup table indexed by (core_smiles, cut_atom_idx).

### M6b — Change-Type Classification

**Goal:** Classify each R-group transformation into medchem-meaningful categories.

**Categories (~8-10):**
- EWG addition/removal (F, Cl, CF3, NO2, CN, SO2, COOH)
- EDG addition/removal (NH2, OH, OMe, NMe2, alkyl)
- Lipophilic change (alkyl chain extension, aromatic ring addition)
- Polar change (add/remove H-bond donor or acceptor)
- Size increase / Size decrease
- Ring gain / Ring loss
- Aromaticity change

**Method:** SMARTS-based classification of R-group fragments. Each transform gets a one-hot (or multi-hot) change-type vector.

### M6c — Interaction Feature Test

**Goal:** Test whether 3D_context × change_type cross-products break the 0.52 ceiling.

**Method:**
1. Build interaction features: outer product of context vector (5-10 dim) × change_type vector (8-10 dim) → 40-100 features
2. Train HGB on interaction features + original deltas
3. Evaluate with same LOO-target NDCG@5 protocol
4. If ceiling breaks → proceed to hybrid model (Approach C)

### M7 — Webapp: SAR Sensitivity Explorer

**Input:** SMILES of compound (or series)
**Pipeline:**
1. Generate 3D conformer (ETKDG)
2. Fragment at all single-cut positions (rdMMPA)
3. For each attachment point:
   a. Compute 3D pharmacophore context
   b. Score all change types via the trained model
   c. Rank change types by predicted cliff probability
**Output:** 2D/3D molecule viewer (Streamlit or Gradio) with:
- Positions colored by SAR sensitivity (red = high cliff probability for any change)
- Click a position → ranked list of change types
- "Position 4: Try EWG (0.72) > lipophilic (0.65) > size increase (0.58)"

**Note:** The model predicts SAR sensitivity (likelihood of a cliff), not direction (increased vs decreased activity). It helps the chemist narrow the scope for the first round of modifications.

---

## 6. Revised Milestone Map (2026-03-18)

| # | Milestone | Status | Model |
|---|---|---|---|
| M1 | Env setup + ChEMBL + EGFR end-to-end | ✅ Done | — |
| M2 | MMP extraction + transformation enrichment | ✅ Done | — |
| M3 | Scale to 50 targets | ✅ Done | — |
| M4 | Feature engineering (2D) | ✅ Done | — |
| M5 | ShinkaEvolve + manual evolution + ceiling analysis | ✅ Done | — |
| M6a | **3D context features at attachment points** | 🔲 NEXT | Sonnet |
| M6b | **Change-type classification of R-groups** | 🔲 | Sonnet |
| M6c | **Interaction feature test (break the 0.52 ceiling?)** | 🔲 | Sonnet (impl) + Opus (interpret) |
| M7 | **Webapp: SAR Sensitivity Explorer** | 🔲 | Sonnet |

## 7. Model Decision Guide

| Task | Model | Rationale |
|---|---|---|
| Writing code from a clear design | Sonnet | Well-scoped, no ambiguity |
| Debugging errors | Sonnet | Straightforward |
| Scientific interpretation of results | Opus | Requires domain reasoning |
| Architectural decisions | Opus | High-stakes design choices |
| Designing ShinkaEvolve fitness functions | Opus | Novel problem formulation |
| Reading/interpreting evolved programs | Opus | Scientific discovery analysis |

---

## 7. Key File Paths

| Path | What |
|---|---|
| `C:\Users\mcucc\Projects\Activity cliffs\` | Project root |
| `D:\Mike project data\Activity cliffs\chembl_36\chembl_36_sqlite\chembl_36.db` | ChEMBL 36 SQLite |
| `src/activity_cliffs/` | Package source |
| `src/activity_cliffs/data/chembl.py` | ChEMBL data loader (fixed for v36 schema) |
| `src/activity_cliffs/data/curation.py` | Activity curation (nM → pActivity) |
| `src/activity_cliffs/data/mmp.py` | MMP extraction (M2) |
| `src/activity_cliffs/features/mmp_features.py` | MMP feature engineering (M4) |
| `src/activity_cliffs/cliffs/miner.py` | Cliff pair miner |
| `src/activity_cliffs/features/featurizer.py` | ECFP4 + descriptors |
| `src/activity_cliffs/models/baselines.py` | LogReg + Ridge baselines |
| `src/activity_cliffs/models/contrastive.py` | MLP contrastive encoder |
| `src/activity_cliffs/series/scaffold.py` | Bemis-Murcko scaffold series |
| `src/activity_cliffs/analysis/visualization.py` | Cliff network + SAR plots |
| `scripts/run_demo.py` | End-to-end demo pipeline |
| `scripts/list_targets.py` | Browse ChEMBL targets by data richness |
| `scripts/extract_mmps.py` | MMP extraction CLI |
| `scripts/prepare_evolve_data.py` | Prepare eval data for ShinkaEvolve (M5) |
| `evolve/initial.py` | Template scoring function (EVOLVE-BLOCK) |
| `evolve/evaluate.py` | Fitness evaluator (NDCG@5, 50 targets) |
| `evolve/run_evo.py` | ShinkaEvolve launcher script |
| `evolve/eval_data/eval_data.npz` | Pre-computed eval data v1 (1.19M rows, 12 feat) |
| `evolve/eval_data/eval_data_v2.npz` | Enriched eval data v2 (1.19M rows, 43 feat + 256 XOR bits) |
| `evolve/ml_ceiling.py` | ML ceiling script v1 (original features) |
| `evolve/ml_ceiling_v2.py` | ML ceiling script v2 (enriched features) |
| `evolve/ANTI_GAMING_PLAN.md` | Overfitting/gaming risk analysis + mitigations |
| `evolve/candidates/` | 17 tested candidate scoring functions (gen1, gen2) |
| `scripts/prepare_evolve_data_v2.py` | Prepare enriched eval data (FP XOR, FG flags, freq) |
| `outputs/demo/CHEMBL203/` | M1 EGFR results |
| `outputs/mmps/all_mmps.parquet` | M3 full MMP corpus (25M rows) |
| `outputs/features/mmp_features.parquet` | M4 feature matrix (25M rows) |
| `outputs/evolve/results/` | ShinkaEvolve output (M5, when run) |
| `PROGRESS_LOG.md` | Step-by-step activity log |

---

## 8. User Profile

- **Domain:** Medicinal chemistry expert — deep SAR intuition, not a coder
- **Hardware:** Windows 11 Pro, RTX 4070
- **Claude plan:** Claude Pro — needs to manage model usage wisely
- **Working style:** Wants exact prompts + model recommendations at each step. Logs progress in PROGRESS_LOG.md. Adaptive plan approach — consider 1-2 steps ahead.
- **Scientific vision:** Build a generalized Topliss tree from massive ChEMBL MMP data, using ShinkaEvolve to discover interpretable rules of which structural changes are most informative for SAR exploration.

---

## 9. Starting a New Session

1. Read this file (CONTINUATION_PLAN.md) and PROGRESS_LOG.md
2. Check "Current Status" in PROGRESS_LOG.md for where we left off
3. Follow the exact prompt listed under "Next Steps"
4. Use the model recommended for that step
5. Update PROGRESS_LOG.md after completing each step
