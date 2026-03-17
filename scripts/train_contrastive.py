from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from activity_cliffs.models import train_contrastive_encoder


app = typer.Typer(add_completion=False, help="Train a contrastive encoder that separates cliff vs non-cliff pairs.")


@app.command()
def main(
    series_parquet: Path = typer.Option(..., "--series", exists=True, help="Path to chembl_series_<target>_<type>.parquet"),
    pairs_parquet: Path = typer.Option(..., "--pairs", exists=True, help="Path to cliff_pairs_<target>_<type>.parquet"),
    outdir: Path = typer.Option(Path("outputs"), "--outdir", help="Output directory."),
    emb_dim: int = typer.Option(128, "--emb-dim", help="Embedding dimension."),
    epochs: int = typer.Option(8, "--epochs", help="Training epochs."),
    batch_size: int = typer.Option(512, "--batch-size", help="Batch size."),
    margin: float = typer.Option(1.0, "--margin", help="Contrastive margin."),
):
    outdir.mkdir(parents=True, exist_ok=True)

    df_series = pd.read_parquet(series_parquet)
    df_pairs = pd.read_parquet(pairs_parquet)
    if len(df_pairs) == 0:
        raise ValueError("Pairs parquet is empty. Mine cliffs first or loosen thresholds.")

    artifacts = train_contrastive_encoder(
        df_series,
        df_pairs,
        emb_dim=emb_dim,
        epochs=epochs,
        batch_size=batch_size,
        margin=margin,
    )

    metrics_path = outdir / "contrastive_metrics.json"
    metrics_path.write_text(json.dumps(artifacts.metrics, indent=2))
    print(f"Wrote: {metrics_path}")
    print(json.dumps(artifacts.metrics, indent=2))


if __name__ == "__main__":
    app()

