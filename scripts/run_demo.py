from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd
import typer

from activity_cliffs.analysis import plot_cliff_network, plot_series_activity
from activity_cliffs.cliffs import mine_activity_cliffs
from activity_cliffs.config import CliffMiningConfig
from activity_cliffs.data.chembl import fetch_target_activities, resolve_chembl_sqlite_path
from activity_cliffs.data.curation import curate_chembl_activities
from activity_cliffs.features import featurize_ecfp4
from activity_cliffs.models import train_baselines_pairwise, train_contrastive_encoder
from activity_cliffs.series import assign_scaffold_series


app = typer.Typer(add_completion=False, help="End-to-end demo: curate → mine cliffs → train → visualize.")


@app.command()
def main(
    targets: List[str] = typer.Option(..., "--targets", help="One or more ChEMBL target_chembl_id values."),
    chembl_sqlite: Optional[Path] = typer.Option(None, "--chembl-sqlite", help="Path to chembl_XX.sqlite."),
    standard_type: str = typer.Option("IC50", "--standard-type", help="ChEMBL standard_type."),
    outdir: Path = typer.Option(Path("outputs/demo"), "--outdir", help="Output directory."),
    sim_min: float = typer.Option(0.85, "--sim-min", help="Minimum similarity for candidate pairs."),
    delta_min: float = typer.Option(1.5, "--delta-min", help="Minimum ΔpActivity for a cliff."),
    min_series_size: int = typer.Option(15, "--min-series-size", help="Only mine scaffolds with at least this many compounds."),
):
    sqlite_path = resolve_chembl_sqlite_path(chembl_sqlite)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = CliffMiningConfig(sim_min=sim_min, delta_pactivity_min=delta_min)

    for target in targets:
        tdir = outdir / target
        tdir.mkdir(parents=True, exist_ok=True)

        raw = fetch_target_activities(sqlite_path, target_chembl_ids=[target], standard_type=standard_type, min_confidence_score=7)
        curated = curate_chembl_activities(raw).df
        series = assign_scaffold_series(curated).df

        # Mine pairs per series
        all_pairs = []
        for sid, df_sid in series.groupby("series_id"):
            if len(df_sid) < min_series_size:
                continue
            fp_res = featurize_ecfp4(df_sid["canonical_smiles"].tolist())
            if fp_res.valid_mask.sum() != len(df_sid):
                df_sid = df_sid.loc[fp_res.valid_mask].reset_index(drop=True)
            pairs = mine_activity_cliffs(df_sid, fp_res.fps, config=cfg, series_id=int(sid)).df
            if len(pairs) > 0:
                all_pairs.append(pairs)

        df_pairs = pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame()
        curated.to_parquet(tdir / "series_curated.parquet", index=False)
        series.to_parquet(tdir / "series_scaffold.parquet", index=False)
        df_pairs.to_parquet(tdir / "cliff_pairs.parquet", index=False)

        if len(df_pairs) == 0:
            print(f"[{target}] No pairs found. Try lower --sim-min or --min-series-size.")
            continue

        # Train baselines + contrastive
        base = train_baselines_pairwise(series, df_pairs)
        (tdir / "baseline_metrics.json").write_text(pd.Series(base.metrics).to_json(indent=2))

        cont = train_contrastive_encoder(series, df_pairs)
        (tdir / "contrastive_metrics.json").write_text(pd.Series(cont.metrics).to_json(indent=2))

        # Visualize the most cliff-rich series
        cliffs = df_pairs[df_pairs["cliff_label"] == 1]
        if len(cliffs) > 0:
            top_sid = int(cliffs["series_id"].value_counts().index[0])
            df_s = series[series["series_id"] == top_sid]
            df_p = df_pairs[df_pairs["series_id"] == top_sid]
            plot_series_activity(df_s, df_p, title=f"{target} series_id={top_sid}", outpath=tdir / "top_series_activity.png")
            plot_cliff_network(df_s, df_p, title=f"{target} series_id={top_sid}", outpath=tdir / "top_series_cliff_network.png")

        print(f"[{target}] Wrote artifacts under: {tdir}")


if __name__ == "__main__":
    app()

