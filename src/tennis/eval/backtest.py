# src/tennis/eval/backtest.py
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from .metrics import log_loss, brier, ece
from typing import List, Tuple, Dict, Any

import lightgbm as lgb

def _df_as_text(df):
    """Prefer markdown if available, otherwise CSV text."""
    try:
        # pandas requires optional 'tabulate' for to_markdown
        return df.to_markdown(index=False)
    except Exception:
        # Fallback that works everywhere
        return df.to_csv(index=False)


def _pick_features(df: pd.DataFrame):
    """
    Select numeric feature columns and the binary target.
    Accepts target named either 'y' or 'target'.
    """
    # Figure out which target column exists
    if "y" in df.columns:
        target_col = "y"
    elif "target" in df.columns:
        target_col = "target"
    else:
        raise KeyError("Expected a target column named 'y' or 'target' in features.")

    # Columns we never feed to the model
    drop_cols = {
        "date", "tour", "surface",
        "winner_name", "loser_name", "tourney_name",
        "match_id_src", "match_id_src_tml",
        "tourney_id_tml", "tourney_name_tml",
        "winner_name_tml", "loser_name_tml",
        target_col,  # drop the detected target from X
    }

    # Keep only numeric columns not in drop list
    num_cols = [
        c for c in df.columns
        if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    X = df[num_cols].copy()
    y = df[target_col].astype(int)
    return X, y



def _year_splits(dates: pd.Series, folds: int = 3) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Build expanding train / single-year validation splits.
    With folds=3 and years 2013..2017, you'll get:
      - train <= 2014-12-31, valid in 2015
      - train <= 2015-12-31, valid in 2016
      - train <= 2016-12-31, valid in 2017
    """
    years = sorted(dates.dt.year.unique().tolist())
    # We need at least folds+1 distinct years (because the first year(s) serve as training)
    # If you have 2013..2017 (5 years) and folds=3 that's perfect.
    out = []
    # Start training through the year before each validation year.
    # E.g. for valid_year=2015 -> train_end=2014 etc.
    for valid_year in years:
        train_end_year = valid_year - 1
        if train_end_year not in years:
            continue
        valid_start = pd.Timestamp(f"{valid_year}-01-01")
        valid_end = pd.Timestamp(f"{valid_year}-12-31")
        out.append((valid_start, valid_end))
    # Keep the last `folds` validation years (most recent)
    out = out[-folds:]
    return out


def _train_lgbm(X_tr, y_tr, X_va, y_va) -> Tuple[lgb.Booster, np.ndarray]:
    train_set = lgb.Dataset(X_tr, label=y_tr)
    valid_set = lgb.Dataset(X_va, label=y_va)
    params = dict(
        objective="binary",
        metric="binary_logloss",
        learning_rate=0.05,
        num_leaves=64,
        min_data_in_leaf=50,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        verbose=-1,
        seed=42,
        feature_fraction_seed=42,
        bagging_seed=42,
        deterministic=True,
        force_row_wise=True,
    )
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=2000,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    preds = booster.predict(X_va, num_iteration=booster.best_iteration)
    return booster, preds



def _train_elo_like(df_tr: pd.DataFrame, df_va: pd.DataFrame) -> np.ndarray:
    """
    Make a very simple Elo-based predictor actually predictive.

    Priority:
      1) If 'elo_exp_a' (expected win prob for player_a) exists, use it and
         optionally calibrate with a 2-parameter sigmoid on the train split.
      2) Else, if we can form 'elo_diff' = rating_a - rating_b from available
         Elo columns, fit a 1-parameter scale and use sigmoid(scale * diff).
      3) Else, fallback to 0.5.
    """
    import numpy as np

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    y_tr = df_tr["target"].to_numpy(dtype=float)

    # -------- Path A: use precomputed expected probability if present --------
    if "elo_exp_a" in df_tr.columns:
        p_raw_tr = np.clip(df_tr["elo_exp_a"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)

        # Optional light calibration: logit(p_cal) = a*logit(p_raw) + b
        # Solve (a, b) with a couple of Newton steps to minimize log loss.
        # Keeps dependencies minimal and is robust.
        logit = lambda q: np.log(q) - np.log1p(-q)
        x = logit(p_raw_tr)
        a, b = 1.0, 0.0  # start close to identity

        for _ in range(5):  # a few iterations are enough
            z = a * x + b
            p = sigmoid(z)
            # Gradient
            grad_a = np.sum((p - y_tr) * x)
            grad_b = np.sum(p - y_tr)
            # Hessian (diagonal approx)
            w = p * (1 - p)
            h_aa = np.sum(w * x * x)
            h_bb = np.sum(w)
            h_ab = np.sum(w * x)
            # 2x2 solve (regularize a hair to avoid singularity)
            H = np.array([[h_aa + 1e-9, h_ab],
                          [h_ab,         h_bb + 1e-9]])
            g = np.array([grad_a, grad_b])
            try:
                da, db = np.linalg.solve(H, g)
                a -= da
                b -= db
            except np.linalg.LinAlgError:
                break

        # Predict on validation
        p_raw_va = np.clip(df_va.get("elo_exp_a", pd.Series(0.5, index=df_va.index)).to_numpy(dtype=float),
                           1e-6, 1 - 1e-6)
        p_cal = sigmoid(a * logit(p_raw_va) + b)
        return np.clip(p_cal, 1e-6, 1 - 1e-6)

    # -------- Path B: construct elo_diff and fit a scale --------
    # Try common rating column pairs in priority order
    rating_pairs = [
        ("elo_a_surface", "elo_b_surface"),
        ("elo_a", "elo_b"),
    ]
    for ra, rb in rating_pairs:
        if ra in df_tr.columns and rb in df_tr.columns and ra in df_va.columns and rb in df_va.columns:
            diff_tr = (df_tr[ra].to_numpy(dtype=float) - df_tr[rb].to_numpy(dtype=float))
            # Line-search a single scale to minimize log loss
            scales = np.linspace(0.005, 0.08, 60)
            best_scale, best_ll = 0.03, np.inf
            for s in scales:
                p = np.clip(sigmoid(s * diff_tr), 1e-6, 1 - 1e-6)
                ll = -np.mean(y_tr * np.log(p) + (1 - y_tr) * np.log(1 - p))
                if ll < best_ll:
                    best_ll, best_scale = ll, s
            # Predict on validation
            diff_va = (df_va[ra].to_numpy(dtype=float) - df_va[rb].to_numpy(dtype=float))
            p_va = np.clip(sigmoid(best_scale * diff_va), 1e-6, 1 - 1e-6)
            return p_va

    # -------- Path C: last resort --------
    return np.full(len(df_va), 0.5, dtype=float)



def walk_forward(
    features: pd.DataFrame,
    folds: int,
    models: List[str],
    tour: str,
    save_dir: str | Path = "data/processed"
) -> Dict[str, Any]:
    """
    Rolling backtest that writes:
      - data/processed/oof_valid.csv   (OOF rows across all validation years)
      - data/processed/backtest_results.csv  (per-fold summary)
    Returns a small dict with paths.
    """
    def _id_col(df: pd.DataFrame) -> str:
        return "match_key" if "match_key" in df.columns else "match_id"

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    df = features.copy()
    # Make sure we have a proper datetime and it's sorted
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Optional tour filtering (supports 'ATP', 'WTA', 'both')
    if tour and tour.lower() in ("atp", "wta"):
        df = df[df["tour"].str.upper() == tour.upper()].copy()
    else:
        df = df.copy()

    # Select features/target
    X_all, y_all = _pick_features(df)

    # Build validation year windows
    val_year_windows = _year_splits(df["date"], folds=folds)
    if not val_year_windows:
        raise RuntimeError("Not enough years to build rolling validation folds.")

    oof_rows = []   # collect per-fold OOF predictions
    rows_summary = []

    for i, (vstart, vend) in enumerate(val_year_windows, start=1):
        # Train set: everything strictly before vstart
        tr_idx = df["date"] < vstart
        va_idx = (df["date"] >= vstart) & (df["date"] <= vend)

        df_tr, df_va = df.loc[tr_idx], df.loc[va_idx]
        X_tr, y_tr = X_all.loc[tr_idx], y_all.loc[tr_idx]
        X_va, y_va = X_all.loc[va_idx], y_all.loc[va_idx]

        if len(df_va) == 0 or len(df_tr) == 0:
            continue  # skip empty folds

        fold_name = f"fold{i}_{vstart.year}"

        fold_preds: Dict[str, np.ndarray] = {}

        if "gbdt" in [m.lower() for m in models]:
            _, p = _train_lgbm(X_tr, y_tr, X_va, y_va)
            fold_preds["gbdt"] = p

        if "elo" in [m.lower() for m in models]:
            p = _train_elo_like(df_tr, df_va)
            fold_preds["elo"] = p

        # Collect OOF rows
        for model_name, preds in fold_preds.items():
            oof_chunk = pd.DataFrame({
                "date": df_va["date"].to_numpy(),
                "match_id": df_va[_id_col(df_va)].to_numpy(),
                "y_true": y_va.to_numpy(),
                "p": preds,
                "model": model_name,
                "fold": fold_name,
            })
            oof_rows.append(oof_chunk)

        fold_metrics_rows = []
        for model_name, preds in fold_preds.items():
            yv = y_va.to_numpy()
            pv = np.clip(preds, 1e-6, 1 - 1e-6)
            fold_metrics_rows.append({
                "fold": fold_name,
                "model": model_name,
                "n": int(len(yv)),
                "log_loss": float(log_loss(yv, pv)),
                "brier": float(brier(yv, pv)),
                "ece": float(ece(yv, pv)),
            })

        # keep a running list for all folds
        rows_summary.extend(fold_metrics_rows)

        # Simple per-fold metric for logging
        def logloss(y, p):
            p = np.clip(p, 1e-6, 1 - 1e-6)
            return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()

        for model_name, preds in fold_preds.items():
            rows_summary.append({
                "fold": fold_name,
                "model": model_name,
                "start": vstart.date().isoformat(),
                "end": vend.date().isoformat(),
                "n_valid": int(va_idx.sum()),
                "logloss": float(logloss(y_va.to_numpy(), preds)),
            })

    # Concatenate and save
    if oof_rows:
        oof = pd.concat(oof_rows, ignore_index=True).sort_values(["date", "model"])
        oof.to_csv(save_dir / "oof_valid.csv", index=False)
    else:
        # Write an empty file so stack/evaluate don't crash, but warn
        oof = pd.DataFrame(columns=["date","match_id","y_true","p","model","fold"])
        oof.to_csv(save_dir / "oof_valid.csv", index=False)

    summary = pd.DataFrame(rows_summary)
    summary.to_csv(save_dir / "backtest_results.csv", index=False)

    # 1) Save fold-level metrics table
    metrics_dir = Path("data/processed")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics_df = pd.DataFrame(rows_summary)
    fold_metrics_path = metrics_dir / "metrics_folds.csv"
    if not fold_metrics_df.empty:
        fold_metrics_df.to_csv(fold_metrics_path, index=False)

    print(f"[fold={fold_name}] " + " | ".join(
        f"{r['model']}: LL={r['log_loss']:.3f}, Br={r['brier']:.3f}, ECE={r['ece']:.3f}"
        for r in fold_metrics_rows
    ))

    # 2) Build overall per-model metrics from the long OOF (one row per model)
    # Expect columns: ["date","match_id","y_true","p","model","fold"]
    oof_long = pd.read_csv(metrics_dir / "oof_valid.csv", parse_dates=["date"])
    overall_rows = []
    for model_name, g in oof_long.groupby("model"):
        yv = g["y_true"].to_numpy()
        pv = np.clip(g["p"].to_numpy(), 1e-6, 1 - 1e-6)
        overall_rows.append({
            "model": model_name,
            "n": int(len(g)),
            "log_loss": float(log_loss(yv, pv)),
            "brier": float(brier(yv, pv)),
            "ece": float(ece(yv, pv)),
        })
    overall_df = pd.DataFrame(overall_rows)
    overall_path = metrics_dir / "metrics_overall.csv"
    if not overall_df.empty:
        overall_df.to_csv(overall_path, index=False)

    # 3) (Optional) also compute an unweighted-mean ensemble metric
    # Pivot to wide by match_id so we ensemble per match across models.
    wide = (
        oof_long
        .pivot_table(index=["date","match_id","y_true"], columns="model", values="p", aggfunc="first")
        .reset_index()
    )
    pcols = [c for c in wide.columns if c not in ("date","match_id","y_true")]
    if pcols:
        ensemble_p = wide[pcols].mean(axis=1).clip(1e-6, 1-1e-6).to_numpy()
        ensemble_y = wide["y_true"].to_numpy()
        overall_rows.append({
            "model": "mean_ensemble",
            "n": int(len(wide)),
            "log_loss": float(log_loss(ensemble_y, ensemble_p)),
            "brier": float(brier(ensemble_y, ensemble_p)),
            "ece": float(ece(ensemble_y, ensemble_p)),
        })
        overall_df = pd.DataFrame(overall_rows)
        overall_df.to_csv(overall_path, index=False)

    # 4) Append a tiny devlog so you can eyeball runs over time
    devlog = metrics_dir / "devlog.md"
    stamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    overall_md = _df_as_text(overall_df.sort_values("log_loss")) if not overall_df.empty else "_no metrics_"
    fold_md = _df_as_text(fold_metrics_df.head(10)) if not fold_metrics_df.empty else "_no fold metrics_"

    devlog_lines = [
        f"## Backtest @ {stamp}",
        f"- folds: {folds}",
        f"- models: {', '.join(models)}",
        "",
        "### Overall metrics",
        overall_md,
        "",
        "### Per-fold metrics (first 10 shown)",
        fold_md,
        "",
    ]
    with devlog.open("a", encoding="utf-8") as f:
        f.write("\n".join(devlog_lines) + "\n\n")

    return {
        "oof_path": str(save_dir / "oof_valid.csv"),
        "summary_path": str(save_dir / "backtest_results.csv"),
        "folds": len(val_year_windows),
    }


