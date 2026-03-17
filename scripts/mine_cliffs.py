from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from activity_cliffs.cliffs import mine_activity_cliffs
from activity_cliffs.config import CliffMiningConfig
from activity_cliffs.data.chembl import fetch_target_activities, resolve_chembl_sqlite_path
from activity_cliffs.data.curation import curate_chembl_activities
from activity_cliffs.features import featurize_ecfp4
from activity_cliffs.series import assign_scaffold_series


app = typer.Typer(add_completion=False, help="Curate ChEMBL and mine activity cliffs for a target.")


@app.command()
def main(
    target: str = typer.Option(..., "--target", help="ChEMBL target_chembl_id (e.g., CHEMBLXXXX)."),
    chembl_sqlite: Optional[Path] = typer.Option(None, "--chembl-sqlite", help="Path to chembl_XX.sqlite."),
    standard_type: str = typer.Option("IC50", "--standard-type", help="ChEMBL standard_type."),
    sim_min: float = typer.Option(0.85, "--sim-min", help="Minimum Tanimoto similarity for candidate pairs."),
    delta_min: float = typer.Option(1.5, "--delta-min", help="Minimum ΔpActivity to label a cliff."),
    min_series_size: int = typer.Option(10, "--min-series-size", help="Only mine scaffolds with at least this many compounds."),
    outdir: Path = typer.Option(Path("outputs"), "--outdir", help="Output directory."),
):
    sqlite_path = resolve_chembl_sqlite_path(chembl_sqlite)
    outdir.mkdir(parents=True, exist_ok=True)

    raw = fetch_target_activities(
        sqlite_path, target_chembl_ids=[target], standard_type=standard_type, min_confidence_score=7
    )
    curated = curate_chembl_activities(raw).df
    series = assign_scaffold_series(curated).df

    # Persist curated + series tables
    curated_path = outdir / f"chembl_curated_{target}_{standard_type}.parquet"
    series_path = outdir / f"chembl_series_{target}_{standard_type}.parquet"
    curated.to_parquet(curated_path, index=False)
    series.to_parquet(series_path, index=False)

    cfg = CliffMiningConfig(sim_min=sim_min, delta_pactivity_min=delta_min)

    all_pairs = []
    for sid, df_sid in series.groupby("series_id"):
        if len(df_sid) < min_series_size:
            continue
        fp_res = featurize_ecfp4(df_sid["canonical_smiles"].tolist())
        # featurizer drops invalid; keep consistent slice
        if fp_res.valid_mask.sum() != len(df_sid):
            df_sid = df_sid.loc[fp_res.valid_mask].reset_index(drop=True)
        pairs = mine_activity_cliffs(df_sid, fp_res.fps, config=cfg, series_id=int(sid)).df
        if len(pairs) > 0:
            pairs.insert(0, "target_chembl_id", target)
            all_pairs.append(pairs)

    out = pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame()
    out_path = outdir / f"cliff_pairs_{target}_{standard_type}.parquet"
    out.to_parquet(out_path, index=False)

    n_pairs = len(out)
    n_cliffs = int(out["cliff_label"].sum()) if n_pairs else 0
    print(f"Wrote: {curated_path}")
    print(f"Wrote: {series_path}")
    print(f"Wrote: {out_path}  (pairs={n_pairs}, cliffs={n_cliffs})")


if __name__ == "__main__":
    app()

