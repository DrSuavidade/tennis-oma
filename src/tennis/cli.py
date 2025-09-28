import typer
from rich import print
from .config import Settings, load_yaml
from .data.ingest_sackmann import load_sackmann_matches
from .data.ingest_tml import load_tml_stats
from .data.ingest_tcb import load_tcb_markets
from .data.xref import build_player_xref, build_tournament_xref
from .data.merge_sources import merge_to_curated
from .data.validate import run_checks
from .data.ingest_matches import load
from .data.ratings import update_elo
from tennis.data.normalize import read_all_sackmann
from .features.builders import add_basic_features, add_rolling_form, add_boxscore_lagged_features, finalize_features, make_symmetric
from .eval.backtest import walk_forward
import pandas as pd
from pathlib import Path
import numpy as np
import json

app = typer.Typer(add_completion=False)

@app.command()
def build_features(asof: str = typer.Option(None, help="Cutoff date YYYY-MM-DD")):
    cfg = load_yaml("configs/base.yaml")
    cur_base = Path(f"data/processed/asof={asof}") if asof else None
    if cur_base and (cur_base / "cur_matches.parquet").exists():
        matches = pd.read_parquet(cur_base / "cur_matches.parquet")
    else:
        # fallback to synth
        from .data.ingest_matches import load
        matches = load("synth", tour=cfg.get("tour", "both"))

    if asof:
        matches = matches[matches["date"] <= pd.to_datetime(asof)]

    # continue as before:
    df = matches.rename(columns={"player_a_id":"player_a_id","player_b_id":"player_b_id"})
    df["score"] = df.get("score","")
    df["retirement"] = df.get("retirement",0)
    df["match_id"] = df.get("match_key", df.index.astype(str))
    df["tour"] = df["tour"]

    # --- NEW: ensure A/B canonical columns exist (A = winner, B = loser) ---
    if "player_a_id" not in df.columns or "player_b_id" not in df.columns:
        # keep originals; just add A/B columns expected by feature/ratings code
        df["player_a_id"] = df["winner_id"]
        df["player_b_id"] = df["loser_id"]
        # optional, helps downstream features if used
        if "winner_name" in df.columns:
            df["a_name"] = df["winner_name"]
        if "loser_name" in df.columns:
            df["b_name"] = df["loser_name"]

    # types & minimal row hygiene for Elo and rolling features
    df = df.dropna(subset=["player_a_id", "player_b_id", "surface", "date"])
    df["player_a_id"] = df["player_a_id"].astype("int64", errors="ignore")
    df["player_b_id"] = df["player_b_id"].astype("int64", errors="ignore")
    # -----------------------------------------------------------------------

    df = add_basic_features(df)
    df = update_elo(df, k_base=32.0, surface_offset=40.0, decay_half_life_days=365)
    df = add_rolling_form(df)
    df = add_boxscore_lagged_features(df)
    feats = finalize_features(df)
    feats = make_symmetric(feats)
    out = Path("data/processed/features.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out, index=False)
    print(f"[green]Saved features -> {out}[/green]")

@app.command()
def train(folds: int = 3, models: str = "elo,gbdt", tour: str = "both"):
    """
    Train with rolling time-based validation and write OOF/backtest files.
    Outputs:
      - data/processed/oof_valid.csv
      - data/processed/backtest_results.csv
    """
    # Load full feature set (don't slice here; walk_forward handles splits)
    feats = pd.read_parquet("data/processed/features.parquet")

    # Optional: keep this filter (walk_forward can also filter by 'tour', but double filtering is harmless)
    if tour in ("ATP", "WTA"):
        feats = feats[feats["tour"] == tour]

    # Parse models into a simple list of names (e.g., ["elo","gbdt"])
    models_list = [m.strip() for m in models.split(",") if m.strip()]

    # Run rolling backtest; it will also write oof/backtest CSVs
    info = walk_forward(features=feats, folds=folds, models=models_list, tour=tour, save_dir="data/processed")

    # Optional: print where files were written
    print(f"[green]Backtest complete. Results saved.[/green]")
    print(f"OOF: {info['oof_path']}")
    print(f"Summary: {info['summary_path']}")


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
    oof = oof[oof["date"] >= pd.to_datetime(since)].copy()

    if {"model", "p", "y_true"}.issubset(oof.columns):
        # --- LONG → WIDE ---
        # pivot probs to p_{model} columns
        wide = (
            oof.pivot_table(
                index=["date", "match_id"],
                columns="model",
                values="p",
                aggfunc="mean",
            )
            .reset_index()
        )
        # rename model columns to p_{model}
        wide.columns = ["date", "match_id"] + [f"p_{c}" for c in wide.columns[2:]]

        # optional market column (if present in long oof)
        if "market_prob_a" in oof.columns:
            mkt = (
                oof.groupby(["date", "match_id"], as_index=False)["market_prob_a"]
                   .mean()
            )
            wide = wide.merge(mkt, on=["date", "match_id"], how="left")

        # build y from y_true
        y_map = (
            oof.groupby(["date", "match_id"])["y_true"]
               .mean().round().astype(int)
        )
        y = y_map.reindex(wide.set_index(["date", "match_id"]).index).to_numpy()

        pcols = [c for c in wide.columns if c.startswith("p_")]
        if not pcols:
            raise SystemExit("No model probability columns found after pivot.")
        p = wide[pcols].mean(axis=1).clip(1e-6, 1-1e-6).to_numpy()

    else:
        # --- WIDE (legacy) ---
        pcols = [c for c in oof.columns if c.startswith("p_")]
        if not pcols:
            raise SystemExit("OOF has neither long (model/p) nor wide (p_*) format.")
        p = oof[pcols].mean(axis=1).clip(1e-6, 1-1e-6).to_numpy()
        y_col = "target" if "target" in oof.columns else (
            "y_true" if "y_true" in oof.columns else None
        )
        if y_col is None:
            raise SystemExit("Could not find target/y_true column in wide OOF.")
        y = oof[y_col].to_numpy()

    print({
        "n": int(len(y)),
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier(y, p)),
        "ece": float(ece(y, p)),
    })


@app.command()
def ingest(
    years: str = typer.Option("2013-2024", help="Year range like 2013-2024"),
    atp: str = typer.Option(None, help="Path to Sackmann ATP"),
    wta: str = typer.Option(None, help="Path to Sackmann WTA"),
    tml: str = typer.Option(None, help="Path to TML-Database"),
    tcb: str = typer.Option(None, help="Path to tennis-crystal-ball"),
):
    cfg = load_yaml("configs/base.yaml")
    y0, y1 = map(int, years.split("-"))

    # Resolve sources (only Sackmann ATP/WTA are needed for normalized matches)
    atp_dir = Path(atp or cfg["sources"]["sackmann_atp"])
    wta_dir = Path(wta or cfg["sources"]["sackmann_wta"])

    # Build one normalized matches table across ATP/WTA (+quals/chall/futures/ITF)
    matches = read_all_sackmann(
        atp_dir=atp_dir,
        wta_dir=wta_dir,
        include_quals_chall=True,
        include_futures_itf=True,
    )

    # Filter to requested year range (by match date)
    matches = matches[
        (matches["date"].dt.year >= y0) & (matches["date"].dt.year <= y1)
    ].copy()

    outdir = Path("data/processed/staging")
    outdir.mkdir(parents=True, exist_ok=True)

    # Write the single, normalized staging file
    matches.to_parquet(outdir / "stg_matches.parquet", index=False)

    # (Optional) Write empty placeholders so older code that expects them won’t crash.
    # They’re safe to keep; freeze will treat them as optional.
    pd.DataFrame().to_parquet(outdir / "stg_tml_stats.parquet", index=False)
    pd.DataFrame().to_parquet(outdir / "stg_tcb_markets.parquet", index=False)

    print("[green]Staging complete.[/green]")


@app.command()
def freeze(asof: str = typer.Option(..., help="YYYY-MM-DD")):
    """Build crosswalks, merge sources, run validation, and freeze curated layers."""
    stg = Path("data/processed/staging")

    # Single, normalized staging table (ATP+WTA+quals/chall/futures/ITF)
    matches_df = pd.read_parquet(stg / "stg_matches.parquet")

    # Backwards-compat aliases for downstream helper signatures
    sack = matches_df

    # Optional extras (markets/stats from other repos). If missing, use empty frames.
    t_tml = stg / "stg_tml_stats.parquet"
    t_tcb = stg / "stg_tcb_markets.parquet"
    tml_df = pd.read_parquet(t_tml) if t_tml.exists() else pd.DataFrame()
    tcb_df = pd.read_parquet(t_tcb) if t_tcb.exists() else pd.DataFrame()

    px = build_player_xref(sack, tml_df, tcb_df)
    tx = build_tournament_xref(sack, tml_df, tcb_df)
    cur_matches, cur_stats, cur_markets = merge_to_curated(sack, tml_df, tcb_df, px, tx)

    issues = run_checks(cur_matches, cur_stats, cur_markets)
    if issues:
        qa_dir = Path("data/processed/qa"); qa_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([issues]).to_json(qa_dir / f"issues_{asof}.json", orient="records", indent=2)
        raise SystemExit(f"Validation failed: {issues}")

    base = Path(f"data/processed/asof={asof}")
    base.mkdir(parents=True, exist_ok=True)
    cur_matches.to_parquet(base / "cur_matches.parquet", index=False)
    cur_stats.to_parquet(base / "cur_stats.parquet", index=False)
    cur_markets.to_parquet(base / "cur_markets.parquet", index=False)
    px.to_parquet(base / "dim_player.parquet", index=False)
    tx.to_parquet(base / "dim_tournament.parquet", index=False)
    print("[green]Freeze complete.[/green]")



if __name__ == "__main__":
    app()
