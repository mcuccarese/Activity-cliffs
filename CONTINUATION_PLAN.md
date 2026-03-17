# Activity Cliffs → Generalized Topliss Tree: Continuation Plan

> **Purpose:** Complete context for picking up this project in a new Claude Code session. Read this file first, then PROGRESS_LOG.md for step-by-step history.

---

## 1. What This Project Is

A research prototype that mines medicinal chemistry SAR data to build a **data-driven, generalized Topliss decision tree**: given any starting compound, recommend which structural modifications to try first to most efficiently discover activity cliffs (steep SAR).

The original 1972 Topliss tree encoded σ (electronic) and π (hydrophobic) Hansch parameters for phenyl substitution as a hand-crafted decision tree. We're learning the equivalent rules from the entire history of medicinal chemistry in ChEMBL, for any scaffold and any changeable motif, using **ShinkaEvolve** (Sakana AI's LLM-driven evolutionary program optimization) to discover interpretable scoring functions.

---

## 2. What's Done (as of 2026-03-17)

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

## 4. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  DATA LAYER (M2-M3)                                         │
│  ChEMBL → curate → MMP extraction → transformation dataset │
│  50-100 targets, 100K-2M (core, R_from, R_to, target, Δp)  │
├─────────────────────────────────────────────────────────────┤
│  FEATURE LAYER (M4)                                         │
│  Per-transformation: Δ-descriptors, attachment env, SMARTS  │
│  Per-context: scaffold rigidity, pharmacophore proximity    │
│  Per-molecule: FPs, descriptors, embeddings, bio-profiles   │
│  Pre-cached as numpy arrays for fast evaluation             │
├─────────────────────────────────────────────────────────────┤
│  DISCOVERY LAYER (M5-M6)                                    │
│  ShinkaEvolve evolves transformation scoring functions      │
│  Fitness: leave-one-target-out NDCG@k                       │
│  Output: interpretable Python scoring functions             │
├─────────────────────────────────────────────────────────────┤
│  APPLICATION LAYER (M7)                                     │
│  Input: starting molecule + target                          │
│  → enumerate modifications → score → rank                   │
│  Output: "Try these changes first"                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Milestone Plan

### M2 — MMP Extraction Module (NEXT — use Sonnet)

**What to build:**
- `src/activity_cliffs/data/mmp.py` — core extraction logic using `rdMMPA.FragmentMol(mol, maxCuts=1, resultsAsMols=False)`
- `scripts/extract_mmps.py` — CLI wrapper with typer

**Algorithm:**
1. For each molecule in curated data, fragment at all single-cut acyclic bond positions
2. Each fragmentation yields (core_smiles, rgroup_smiles) with `[*:1]` attachment points
3. Group all fragments by canonical core_smiles
4. Within each group, every pair of molecules sharing a core is an MMP
5. Record: core, rgroup_from, rgroup_to, transform_smarts (rgroup_from>>rgroup_to), ΔpActivity
6. Cap group size at 200 to prevent O(n²) explosion

**Output columns:** target_chembl_id, mol_from, mol_to, smiles_from, smiles_to, core_smiles, rgroup_from, rgroup_to, transform_smarts, pActivity_from, pActivity_to, delta_pActivity, abs_delta_pActivity

**Validate:** Run on EGFR (CHEMBL203). Expect >2K MMP rows.

**Exact prompt for Sonnet:**
```
Write the MMP extraction module at src/activity_cliffs/data/mmp.py using rdMMPA.FragmentMol for single-cut fragmentation, plus the CLI script at scripts/extract_mmps.py. Follow the design from the progress log: fragment every molecule, group by canonical core, pair within groups, output parquet with columns for core, rgroups, transform_smarts, and delta_pActivity. Cap group size at 200 to control pair explosion. Then run it on EGFR (CHEMBL203) using the ChEMBL database at "D:\Mike project data\Activity cliffs\chembl_36\chembl_36_sqlite\chembl_36.db" to validate.
```

### M2b — Transformation Enrichment Analysis (Opus)

Before scaling, analyze EGFR MMPs: which transformations most frequently cause large |ΔpActivity|? This is the data-driven Topliss analog — potentially publishable even without ML.

**Prompt for Opus:**
```
The EGFR MMP extraction is done. Analyze the transformation enrichment: for each unique transform_smarts, compute the mean and median abs_delta_pActivity, count, and fraction of "cliff-inducing" transforms (abs_delta >= 1.5). Show me the top 20 most informative transformations and interpret what physicochemical/pharmacological patterns they reveal. Compare to Topliss's original σ/π axes.
```

### M3 — Scale to 50 Targets (Sonnet)

Extract MMPs from the top 50 ChEMBL targets (those with >500 IC50 compounds). Combine into `outputs/mmps/all_mmps.parquet`. This is the ShinkaEvolve training corpus.

**Target selection:** Already have the top 30 from `list_targets --top 30`. Extend to 50. Filter to Homo sapiens only. Key targets include:
- EGFR (CHEMBL203) — already done
- BTK (CHEMBL5251), JAK2 (CHEMBL2971), VEGFR2 (CHEMBL279) — kinases
- BACE1 (CHEMBL4822) — protease (non-kinase validation)
- BRD4 (CHEMBL1163125) — bromodomain
- HDAC1 (CHEMBL325) — zinc enzyme

### M4 — Feature Engineering (Sonnet + Opus for design)

Pre-compute and cache features for every molecule and transformation in the corpus:

**Transformation features:**
- Δ in Hansch-like parameters (ΔLogP, ΔTPSA, ΔMW, ΔHBDonors, ΔHBAcceptors)
- Size change (Δ heavy atom count)
- R-group fingerprints (Morgan FP of each R-group fragment)

**Context features:**
- Attachment point atom environment (Morgan substructure hash at cut bond)
- Local scaffold rigidity (rotatable bonds near attachment)
- Pharmacophore features near attachment point

**Global molecule features:**
- ECFP4, MACCS keys, physicochemical descriptors
- (Optional) Mol2Vec embeddings, predicted bioactivity profiles

Store as numpy arrays indexed by (target, molregno) for fast ShinkaEvolve evaluation.

### M5 — ShinkaEvolve Integration (Opus for design, Sonnet for implementation)

**Install:** `pip install shinkaevolve` (Apache 2.0, ICLR 2026)

**Template function to evolve:**
```python
def score_transformation(
    context_smiles: str,      # shared scaffold with [*:1]
    frag_from: str,           # R-group being removed
    frag_to: str,             # R-group being added
    attachment_env: dict,     # pre-computed atom environment features
    mol_descriptors: dict,    # pre-computed molecular descriptors
) -> float:
    """Score how likely this transformation is to cause a large
    activity change. Higher = more informative modification.
    Available: rdkit, numpy, scipy, sklearn."""
```

**Fitness evaluator:**
- For each held-out target (leave-one-target-out):
  - For each starting molecule with ≥3 known transformations:
    - Score all transformations with the evolved function
    - Measure NDCG@5 (did the top-ranked ones actually cause the biggest |ΔpActivity|?)
- Return mean NDCG@5 across all held-out targets

**Key ShinkaEvolve config:**
- Pre-cache all features so evaluation is microseconds per transformation
- Use ~200-300 generations (ShinkaEvolve is sample-efficient)
- Run on CPU (no GPU needed for evaluation)

### M6 — Analysis & Iteration (Opus)

Read the top-scoring evolved programs. Interpret what features they use. Compare to Topliss. Incorporate domain insight (user is a medchem expert). Re-evolve with refined constraints.

### M7 — Recommendation Interface (Sonnet)

Notebook or simple script:
1. User provides SMILES + target
2. System enumerates feasible modifications (BRICS cuts, functional group swaps)
3. Scores each with the best evolved function
4. Returns ranked list of recommended modifications

---

## 6. Model Decision Guide

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
| `src/activity_cliffs/data/mmp.py` | **TO BUILD** — MMP extraction |
| `src/activity_cliffs/cliffs/miner.py` | Cliff pair miner |
| `src/activity_cliffs/features/featurizer.py` | ECFP4 + descriptors |
| `src/activity_cliffs/models/baselines.py` | LogReg + Ridge baselines |
| `src/activity_cliffs/models/contrastive.py` | MLP contrastive encoder |
| `src/activity_cliffs/series/scaffold.py` | Bemis-Murcko scaffold series |
| `src/activity_cliffs/analysis/visualization.py` | Cliff network + SAR plots |
| `scripts/run_demo.py` | End-to-end demo pipeline |
| `scripts/list_targets.py` | Browse ChEMBL targets by data richness |
| `scripts/extract_mmps.py` | **TO BUILD** — MMP extraction CLI |
| `outputs/demo/CHEMBL203/` | M1 EGFR results |
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
