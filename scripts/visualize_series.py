from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from activity_cliffs.analysis import draw_cliff_pair, plot_cliff_network, plot_series_activity


app = typer.Typer(add_completion=False, help="Visualize mined cliffs for the most cliff-rich series.")


@app.command()
def main(
    series_parquet: Path = typer.Option(..., "--series", exists=True, help="chembl_series_<target>_<type>.parquet"),
    pairs_parquet: Path = typer.Option(..., "--pairs", exists=True, help="cliff_pairs_<target>_<type>.parquet"),
    outdir: Path = typer.Option(Path("outputs/viz"), "--outdir", help="Output directory for plots."),
    top_series: int = typer.Option(5, "--top-series", help="How many series to visualize (by # cliffs)."),
    top_pairs: int = typer.Option(3, "--top-pairs", help="How many example cliff pairs to draw per series."),
):
    outdir.mkdir(parents=True, exist_ok=True)

    df_series = pd.read_parquet(series_parquet)
    df_pairs = pd.read_parquet(pairs_parquet)
    if len(df_pairs) == 0:
        raise ValueError("Pairs parquet is empty. Mine cliffs first.")

    cliffs = df_pairs[df_pairs["cliff_label"] == 1].copy()
    if "series_id" not in cliffs.columns:
        raise ValueError("Expected `series_id` in pairs parquet.")

    series_rank = (
        cliffs.groupby("series_id", as_index=False)
        .agg(n_cliffs=("cliff_label", "size"), max_delta=("delta_pActivity", "max"))
        .sort_values(["n_cliffs", "max_delta"], ascending=[False, False])
        .head(top_series)
    )

    smiles_by_mol = dict(zip(df_series["molregno"].astype(int), df_series["canonical_smiles"].astype(str)))

    for row in series_rank.itertuples(index=False):
        sid = int(row.series_id)
        df_s = df_series[df_series["series_id"] == sid].copy()
        df_p = df_pairs[df_pairs["series_id"] == sid].copy()

        title = f"series_id={sid}  (n_cmpds={len(df_s)})"
        plot_series_activity(df_s, df_p, title=title, outpath=outdir / f"series_{sid}_activity.png")
        plot_cliff_network(df_s, df_p, title=title, outpath=outdir / f"series_{sid}_cliff_network.png", only_cliffs=True)

        examples = (
            df_p[df_p["cliff_label"] == 1].sort_values("delta_pActivity", ascending=False).head(top_pairs)
        )
        for k, r in enumerate(examples.itertuples(index=False), start=1):
            smi_a = smiles_by_mol.get(int(r.mol_i), "")
            smi_b = smiles_by_mol.get(int(r.mol_j), "")
            if not smi_a or not smi_b:
                continue
            draw_cliff_pair(
                smi_a,
                smi_b,
                legend_a=f"mol_i={int(r.mol_i)}  pΔ={float(r.delta_pActivity):.2f}",
                legend_b=f"mol_j={int(r.mol_j)}  sim={float(r.sim):.2f}",
                outpath=outdir / f"series_{sid}_pair_{k}.png",
            )

    print(f"Wrote visualizations under: {outdir}")


if __name__ == "__main__":
    app()

