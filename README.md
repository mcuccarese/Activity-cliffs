# Activity Cliffs Prototype

Research prototype to mine **activity cliffs** from public medicinal chemistry data (starting with **ChEMBL**), train baseline models to predict cliff behavior, and visualize cliff-sensitive regions for SAR analysis.

## What this repo does

- **Curate** ChEMBL bioactivity data for one or more targets
- **Mine** activity-cliff pairs using similarity + \(\Delta\) activity thresholds
- **Train** baseline models (fingerprints → cliff vs non-cliff; \(\Delta\) activity)
- **Prototype** a contrastive embedding model for “cliffness” separation
- **Visualize** cliffs as SAR plots, cliff networks, and change-localization

## Quickstart (Windows)

### 1) Create environment (recommended: conda/mamba)

Create the environment from [`environment.yml`](environment.yml) (added in this repo) and activate it.

### 2) Obtain ChEMBL SQLite

Download a ChEMBL release and locate the SQLite DB file (usually named like `chembl_XX.sqlite`).

Set an environment variable pointing to it:

- PowerShell:
  - `setx CHEMBL_SQLITE_PATH "C:\\path\\to\\chembl_XX.sqlite"`

### 3) List candidate targets (high-data)

Run:

- `python -m scripts.list_targets --top 25`

Then pick 1–3 targets (the workflow defaults to a small starter set if you don’t specify).

### 4) Run a small end-to-end demo

- `python -m scripts.run_demo --targets CHEMBL_TARGET_ID`

Outputs (under `outputs/`) include curated tables, mined cliff pairs, baseline model metrics, and plots.

## Repo layout

- `src/`: library code (data loading, featurization, cliff mining, modeling, visualization)
- `scripts/`: runnable entrypoints
- `notebooks/`: exploratory notebooks
- `outputs/`: generated artifacts (ignored from source control by default)

