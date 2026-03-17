# Activity Cliffs → Generalized Topliss Tree

## Start of Session
1. Read `CONTINUATION_PLAN.md` for full project context and architecture
2. Read `PROGRESS_LOG.md` for current status and next steps
3. Follow the exact prompt and model listed under "Next Steps"
4. Update PROGRESS_LOG.md after each completed step

## Build & Run
- Conda env: `conda activate activity-cliffs`
- Install: `pip install -e .`
- Python: use `c:/Users/mcucc/miniforge3/envs/activity-cliffs/python` if conda isn't active in shell
- ChEMBL SQLite: `D:\Mike project data\Activity cliffs\chembl_36\chembl_36_sqlite\chembl_36.db`
- Pass `--chembl-sqlite "D:\Mike project data\Activity cliffs\chembl_36\chembl_36_sqlite\chembl_36.db"` to scripts (env var may not persist across terminal restarts)

## Code Style
- Python 3.11, type hints on public functions
- RDKit fingerprints: use `rdFingerprintGenerator.GetMorganGenerator` (NOT deprecated `AllChem.GetMorganFingerprintAsBitVect`)
- ChEMBL 36 schema: column is `td.chembl_id` (not `td.target_chembl_id`)
- Parquet for all data artifacts, JSON for metrics

## User Context
- Domain expert in medicinal chemistry, not a coder
- Wants exact prompts and model recommendations at each step
- Manages Claude Pro usage — use Sonnet by default, Opus for science/architecture only
- Hardware: Windows 11 Pro, RTX 4070

## Project Architecture
- Package: `src/activity_cliffs/`
- Scripts: `scripts/` (CLI entry points via typer)
- Outputs: `outputs/` (generated artifacts, not tracked in git)
- Config: `src/activity_cliffs/config.py` (dataclasses)

## Key Decisions
- MMPs (matched molecular pairs) are the core data unit, not similarity-based cliff pairs
- ShinkaEvolve discovers interpretable scoring functions via evolutionary optimization
- Leave-one-target-out NDCG@k is the fitness metric for generalization
- Target-specific models are acceptable; the Topliss tree aims for general rules
