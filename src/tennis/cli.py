import typer
from rich import print
from .config import Settings, load_yaml
from .data.ingest_matches import load
from .data.ratings import update_elo
from .features.builders import add_basic_features, add_rolling_form, finalize_features
from .eval.backtest import walk_forward
import pandas as pd
from pathlib import Path
import numpy as np
import json

app = typer.Typer(add_completion=False)

@app.command()
def build_features(asof: str = typer.Option(None, help="Cutoff date YYYY-MM-DD")):
    cfg = load_yaml("configs/base.yaml")
    df = load("synth", tour=cfg.get("tour", "both"))
    if asof:
        df = df[df["date"] <= pd.to_datetime(asof)]
    # basic target
    df = add_basic_features(df)
    # elo
    df = update_elo(df, k_base=32.0, surface_offset=40.0, decay_half_life_days=365)
    # rolling
    df = add_rolling_form(df)
    # finalize
    feats = finalize_features(df)
    out = Path("data/processed/features.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out, index=False)
    print(f"[green]Saved features -> {out}[/green]")

@app.command()
def train(folds: int = 3, models: str = "elo,gbdt", tour: str = "both"):
    cv_cfg = load_yaml("configs/cv.yaml")["cv"]
    cv_cfg["n_folds"] = folds
    m_cfg = load_yaml("configs/models.yaml")["models"]

    feats = pd.read_parquet("data/processed/features.parquet")
    if tour in ("ATP","WTA"):
        feats = feats[feats["tour"] == tour]

    sel_models = {k: m_cfg[k] for k in models.split(",")}
    results, oof = walk_forward(feats, sel_models, cv_cfg)
    results.to_csv("data/processed/backtest_results.csv", index=False)
    oof.to_csv("data/processed/oof_valid.csv", index=False)
    print("[green]Backtest complete. Results saved.[/green]")

@app.command()
def stack(since: str = "2018-01-01", use_market: bool = False):
    from .ensemble.stacker import OOFStacker
    oof = pd.read_csv("data/processed/oof_valid.csv", parse_dates=["date"])
    min_d, max_d = oof["date"].min(), oof["date"].max()
    oof = oof[oof["date"] >= pd.to_datetime(since)]
    if oof.empty:
        raise SystemExit(
            f"No OOF rows on/after {since}. Available date range: {min_d.date()} → {max_d.date()}."
        )
    stk = OOFStacker(use_market=use_market)
    stk.fit(oof)
    # Save coefficients
    import joblib, numpy as np
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    joblib.dump(stk, "data/processed/stacker.joblib")
    print("[green]Stacker trained and saved.[/green]")

@app.command()
def evaluate(since: str = "2018-01-01"):
    import pandas as pd
    from .eval.metrics import log_loss, brier, ece
    oof = pd.read_csv("data/processed/oof_valid.csv", parse_dates=["date"])
    oof = oof[oof["date"] >= pd.to_datetime(since)]
    # simple average for demo
    pcols = [c for c in oof.columns if c.startswith("p_")]
    p = oof[pcols].mean(axis=1).clip(1e-6,1-1e-6).to_numpy()
    y = oof["target"].to_numpy()
    print({
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier(y, p)),
        "ece": float(ece(y, p))
    })

if __name__ == "__main__":
    app()
