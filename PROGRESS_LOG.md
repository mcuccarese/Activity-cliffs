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
| **3D context features** | `src/activity_cliffs/features/context_3d.py` | ✅ M6a — done |
| **3D context lookup** | `outputs/features/context_3d.parquet` | ✅ 104,346 cores × 9 features, 3.1 MB |
| **Change-type classification** | `src/activity_cliffs/features/change_type.py` | ✅ M6b — done |
| **R-group property lookup** | `outputs/features/rgroup_props.parquet` | ✅ 376,299 R-groups × 11 features, 9.0 MB |
| **Interaction feature test** | `evolve/ml_ceiling_v3.py` | ✅ M6c — done (ceiling holds) |
| **Eval data v3** | `evolve/eval_data/eval_data_v3.npz` | ✅ 1.19M rows × 127 features, 85.6 MB |
| **Position-level data** | `scripts/prepare_position_data.py` | ✅ M7a — done |
| **Position eval data** | `evolve/eval_data/position_data.npz` | ✅ 598K rows × 11 features, 4.3 MB |
| **Position ceiling test** | `evolve/position_ceiling.py` | ✅ M7a — NDCG@3=0.964, Hit@1=57% |
| **Pharmacophore homology** | `scripts/pharmacophore_homology.py` | ✅ M7b — no improvement (targets too similar, r>0.91) |
| **SAR profiles + clusters** | `outputs/pharmacophore_homology/` | ✅ Heatmap, dendrogram, cluster assignments, results JSON |
| **Final position model** | `webapp/model/position_hgb.pkl` | ✅ M7c — HGB trained on all 598K rows |
| **SAR Sensitivity Explorer** | `webapp/app.py` | ✅ M7c — Streamlit webapp with molecule coloring |
| **Prediction pipeline** | `webapp/predict.py` | ✅ M7c — SMILES → fragment → 3D features → predict |
| **Change-type training** | `scripts/train_change_type_model.py` | ✅ M9 — LOO Spearman 0.268 ± 0.068 |
| **Change-type model** | `webapp/model/change_type_hgb.pkl` | ✅ M9 — HGB on 5M MMPs (1 MB) |
| **Change-type webapp** | `webapp/app.py` (3-col detail panel) | ✅ M9 — orange bar chart, per-position recommendations |

---

## Milestone Map (revised 2026-03-19 — position-level reframe)

| # | Milestone | Status | Model |
|---|---|---|---|
| M1 | Env setup + ChEMBL + EGFR end-to-end run | ✅ Done | — |
| M2 | MMP extraction module + validate on EGFR | ✅ Done | — |
| M3 | Scale MMP extraction to 50 targets | ✅ Done | — |
| M4 | Feature engineering (2D) | ✅ Done | — |
| M5 | ShinkaEvolve + manual evolution + ML ceiling analysis | ✅ Done | — |
| M6a | **3D context features at attachment points** | ✅ Done | Sonnet |
| M6b | **Change-type classification of R-groups** | ✅ Done | Sonnet |
| M6c | **Interaction feature test (break 0.52 ceiling?)** | ✅ Done (ceiling holds) | Sonnet + Opus |
| M7a | **Position-level reframe + ceiling test** | ✅ Done (NDCG@3=0.964, Hit@1=57%) | Opus |
| M7b | **Pharmacophore homology grouping** | ✅ Done (no improvement — targets are too similar) | Opus |
| M7c | **Webapp: SAR Sensitivity Explorer** | ✅ Done | Sonnet |
| M8 | **Interactive explainability: clickable bonds, structures, ChEMBL links** | ✅ Done | Sonnet |
| M9 | **Change-type recommendations: Topliss-style "start here" per position** | ✅ Done (LOO Spearman 0.268 ± 0.068) | Sonnet + Opus |

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
| 2026-03-18 | M6a: 3D pharmacophore context features computed | Built `context_3d.py` module + `compute_3d_context.py` CLI. For each of 104,346 unique cores: replaced [*:1] with H, ETKDG conformer + MMFF, computed 9 features at attachment atom (donor/acceptor/hydrophobic/aromatic counts within 4Å, SASA, Gasteiger charge, rotatable bonds within 2 bonds, aromatic flag, heavy atom density). **0 parse failures, 99.99% got 3D features** (13 topological-only fallback). 53 min on single CPU (32.8 cores/s). Key stats: 42% aromatic attachment, mean charge -0.077, mean 6.4 heavy atoms within 4Å. Bugs fixed during dev: Gasteiger NaN from dummy atoms (compute on capped mol), GetShortestPath self-loop crash, GetTotalNumHs=0 after AddHs (count bonded H neighbors). |
| 2026-03-18 | M6b: R-group change-type classification | Built `change_type.py` module + `compute_change_types.py` CLI. 11-dim property vector per R-group: EWG/EDG detection (SMARTS), HBD/HBA counts, LogP, heavy atoms, rings, aromatic rings, fsp3. **376,299 unique R-groups classified in 3.7 min** (1,679 rg/s), 0 failures → `outputs/features/rgroup_props.parquet` (9.0 MB). Validated on medchem examples: F→OMe correctly shows EWG→EDG swap; Me→Ph captures ring/aromaticity gain; Ph→cHex shows aromaticity loss + sp3 gain. Cliff vs non-cliff deltas: cliffs have larger delta_heavy_atoms (+0.60), delta_ewg_count (+0.10), delta_lipophilicity (+0.08), delta_n_rings (+0.10) — consistent with M2b finding that bigger structural changes drive cliffs. |
| 2026-03-18 | M6c: Interaction feature test — **ceiling NOT broken** | Built `prepare_evolve_data_v3.py` (joins 2D deltas + 3D context + change-type + interactions -> 127 features, 1.19M rows, 85.6 MB) and `ml_ceiling_v3.py` (14 experiments). **All combinations <= 0.5170, matching the original 8-feature baseline.** 3D context alone = 0.4124 (worse than random -- no within-group variance as predicted). Change-type deltas alone = 0.5108 (competitive but redundant with property deltas). Context x change_type interactions = 0.5032 (HGB already learns interactions natively). Adding features to baseline slightly hurts (0.5126-0.5155). Deeper HGB (500 iter, depth 8) = 0.5142. **Conclusion: the ~0.52 ceiling is a fundamental limit of target-agnostic ligand descriptors.** What makes a specific R-group swap cause a cliff depends on the protein binding pocket geometry, which no ligand-only feature captures. The path forward requires either target-specific models or protein-side features. |
| 2026-03-19 | M7a: Position-level reframe — **breakthrough result** | Reframed question from "which R-group swap causes the biggest cliff?" (transformation-level) to "which position on a molecule is most SAR-sensitive?" (position-level). Aggregated 25M MMPs to 598K (mol_from, position, target) rows across 50 targets, 46K unique cores, 111K molecules. Features: 9 3D context + 2 core topology = 11 features. **Key finding: 3D context features that scored 0.41 (below random) at transformation level now score 0.946 NDCG@3 at position level.** Best model (HGB, 11 features): NDCG@3=0.964, Hit@1=57% (2.3x random 25%), Spearman=0.607. Position sensitivity generalizes across targets (leave-one-target-out). Within-molecule position range = 0.50 pActivity units (meaningful variance). Strongest signals: core_n_heavy (r=-0.51), core_n_rings (r=-0.45), n_heavy_4A (r=-0.38), n_aromatic_4A (r=-0.27). Interpretation: simpler, less crowded, less constrained positions are more SAR-sensitive -- the R-group represents a larger fraction of the binding interaction. |
| 2026-03-19 | M7b: Pharmacophore homology grouping — **no improvement (informative negative)** | Built 28-dim SAR profile per target (14 change-type categories × {cliff_rate, mean_|Δp|}). **Targets are extremely similar:** pairwise Pearson mean=0.983, min=0.912, all >0.9. Hierarchical clustering (Ward linkage) tested k=5-8. **Experiment A (global + cluster_id):** best k=5 NDCG@3=0.9597 vs baseline 0.9593 — within noise (+0.0004). **Experiment A2 (one-hot clusters):** best k=7 NDCG@3=0.9601 (+0.0008). **Experiment B (within-cluster separate models):** all HURT performance (k=5: 0.9549, k=8: 0.9518) — less training data, no compensating signal. **Experiment C (SAR profile as features):** 0.9593 (identical to baseline). **Conclusion:** Position-level SAR sensitivity is governed by local 3D pharmacophore context, not target identity. The rules are truly general — all 50 targets respond to structural changes the same way at the position level. No target grouping needed; proceed directly to webapp with the target-agnostic model. |
| 2026-03-19 | M7c: SAR Sensitivity Explorer webapp — **complete** | Built Streamlit webapp (`webapp/app.py`) with prediction pipeline (`webapp/predict.py`). Trained final HGB model on all 598K position rows (1 MB pickle). Pipeline: SMILES → find all fragmentable bonds → fragment → compute 3D pharmacophore context (9 features) + core topology (2 features) → HGB predict sensitivity → rank positions. Molecule visualization: 2D SVG with blue→red atom/bond coloring + rank labels (#1, #2...). Position ranking with expandable feature breakdowns per position. **Permutation feature importances:** core_n_heavy (45.4%), gasteiger_charge (14.1%), SASA (10.7%), core_n_rings (8.5%), n_aromatic_4A (4.6%), n_heavy_4A (4.6%). **Medchem validation:** Imatinib correctly identifies piperazine arm > methyl > hinge binder; Erlotinib highlights aniline bridge > acetylene tail > methoxyethoxy chains; Celecoxib flags pyrazole-tolyl junction; Diclofenac prioritizes chlorine positions; small molecules (aniline, biphenyl) show appropriately high sensitivity. Model shows larger cores → lower sensitivity (correct: R-group is smaller fraction of binding). 8 example drugs included. Run: `streamlit run webapp/app.py`. |

| 2026-03-20 | M9: Change-type recommendations — **complete** | Built `scripts/train_change_type_model.py` (20D input: 9D pharmacophore context + 11D Δ R-group properties → predict |ΔpActivity|). Trained on 5M stratified-sampled MMPs across 50 targets in 21s. **LOO-target Spearman: mean=0.268 ± 0.068, min=0.142, max=0.417.** All 50 targets positively correlated — model never anti-predicts. This is a transformation-level prediction (harder than position-level: individual MMP cliff magnitude, not just average sensitivity). Inference: ±1σ probing along 11 Δ-prop axes, rank by max predicted |Δ|. **Medchem sanity checks passed:** (1) Imatinib: 9/10 unique rankings across positions; methyl→DFG loop flags H-bond acceptor change, pyrimidine hinge flags aromatic ring change. (2) Diclofenac: **lipophilicity change** ranked #1 at chlorine positions (score 1.02) and carboxylate (1.03) — the textbook cliff-forming axes for NSAIDs. (3) Celecoxib: **EDG count change** ranked #1 at tolyl-pyrazole junction — first time EDG beats size, correct for the COX-2 selectivity pocket. Sanity check on archetypal contexts: hydrophobic pocket → EDG count (0.68); donor-rich → size (0.58); crowded site → size (0.73). Webapp: 3-column detail panel (SHAP | change types | evidence), orange bar chart via `_render_change_type_recs()`. Model: `webapp/model/change_type_hgb.pkl` (1 MB), metadata: `webapp/model/change_type_meta.json`. |

---

## Current Status

**2026-03-20** — **M9 complete.** Change-type recommendation model trained and integrated into the webapp. Each position now shows not only HOW sensitive it is (position model, M7) but also WHAT TYPE of structural modification is most likely to cause a large activity swing (change-type model, M9). The webapp has a 3-column detail panel: SHAP attributions | change-type recommendations | real-world evidence. LOO-target Spearman = 0.268 ± 0.068 — modest but statistically significant for transformation-level prediction without protein features. Medchem validation on 3 drugs shows genuine context-dependent variation: Imatinib (size-dominated), Diclofenac (lipophilicity at chlorines/carboxylate), Celecoxib (EDG at tolyl junction).

**Potential future work:**
- Deploy to a cloud service for team access
- Publication-quality figures / validation against published SAR studies
- Protein-side features to break the transformation-level ceiling
- Specific R-group suggestions (not just property type) via retrieval from ChEMBL MMP database

---

## Next Steps

### M9 — ✅ DONE (2026-03-20): Change-Type Recommendations

### Previous steps (done)

- M7a — ✅ DONE (2026-03-19): Position-level reframe validated
- M7b — ✅ DONE (2026-03-19): Pharmacophore homology grouping — informative negative
- M7c — ✅ DONE (2026-03-19): SAR Sensitivity Explorer webapp

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
