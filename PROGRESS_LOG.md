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
| **Real data run** | Any `outputs/` artifacts | ❌ Never executed |

---

## Milestone Map (revised 2026-03-16 after full code review)

| # | Milestone | Status | Est. sessions |
|---|---|---|---|
| M1 | Env setup + ChEMBL data + target selection | 🔄 In progress | current |
| M2 | First end-to-end run on 1 target, review outputs | 🔲 Not started | 1 |
| M3 | Add molecule-level cliff propensity (k-NN scoring) | 🔲 Not started | 1 |
| M4 | Add MMP analysis on cliff pairs (most actionable feature) | 🔲 Not started | 1-2 |
| M5 | Contrastive model training + checkpoint saving | 🔲 Not started | 1 |
| M6 | Atom attribution + fragment enrichment analysis | 🔲 Not started | 2-3 |

---

## Completed Steps

| Date | Step | Result |
|---|---|---|
| 2026-03-16 | Installed Miniforge3 + initialized conda for PowerShell | conda works in PowerShell |
| 2026-03-16 | `conda env create -f environment.yml` + `pip install -e .` | `activity-cliffs` env active, package installed |
| 2026-03-16 | Smoke test | PyTorch + RDKit + activity_cliffs all import cleanly |

---

## Current Status

**2026-03-16** — M1 env portion complete. **Next: download ChEMBL SQLite (~4 GB).** CUDA not yet verified.

---

## Next Steps

### ~~Step 1 — Create the conda environment~~ ✅ DONE 2026-03-16

---

### Step 2 — Download ChEMBL and point the project at it
**When:** After Step 1 succeeds.
**Model:** None — this is a download task.

1. Go to [https://chembl.gitbook.io/chembl-interface-documentation/downloads](https://chembl.gitbook.io/chembl-interface-documentation/downloads) and download the **SQLite** version of the latest ChEMBL release (e.g., `chembl_34_sqlite.tar.gz`, ~4 GB compressed).
2. Extract it — you'll get a file like `chembl_34.db` or `chembl_34.sqlite`.
3. In PowerShell (run once, persists across sessions):
   ```powershell
   setx CHEMBL_SQLITE_PATH "C:\path\to\chembl_34.sqlite"
   ```
   Replace the path with where you actually saved the file. Close and reopen your terminal after running this.

**Success signal:** Running `echo $env:CHEMBL_SQLITE_PATH` in PowerShell prints the path.

---

### Step 3 (coming after Steps 1–2) — Browse targets and pick one
**Model:** **Sonnet**
**Prompt to use:**
```
I've set up the conda environment and downloaded ChEMBL. I want to run the list_targets script to see the best targets for activity cliff mining. The project is at C:\Users\mcucc\Projects\Activity cliffs.

Please show me the exact command to run and explain what the output columns mean so I can pick a good starting target. I'm looking for a human target with at least a few hundred IC50 measurements, ideally a kinase or GPCR.
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

## Known Gaps (prioritized by impact for medchem)

1. **MMP analysis** — Not implemented. The single highest-value addition. Would show *which specific R-group swaps cause cliffs* across series. RDKit `rdMMPA` supports this.
2. **Molecule-level cliff propensity** — Not implemented. Would let you score individual candidate SMILES for cliff risk via k-NN over mined data.
3. **Model persistence** — Contrastive encoder trains but never saves weights. Can't reload to score new compounds.
4. **Atom-level attribution** — Not implemented. Integrated gradients on the MLP would show which atoms drive cliff predictions.
5. **MLP-only, no GNN** — Fine for prototype. GNN (PyTorch Geometric) would enable richer atom attribution but adds complexity.
6. **No Streamlit app** — Notebooks are the right interface for now.
