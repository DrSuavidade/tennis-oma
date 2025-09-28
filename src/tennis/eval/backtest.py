# src/tennis/eval/backtest.py
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any

import lightgbm as lgb


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
    Very simple Elo probability based on precomputed elo features if present.
    If you have an elo_diff feature, convert to prob via logistic; otherwise fallback to 0.5.
    """
    if "elo_diff" in df_tr.columns:
        # Fit a one-parameter scale by simple calibration (logistic regression with fixed intercept=0)
        # prob = 1 / (1 + exp(-scale * elo_diff))
        # Solve scale with a quick line-search; keep it simple and robust.

        def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))

        X = df_tr["elo_diff"].to_numpy()
        y = df_tr["y"].to_numpy()
        scales = np.linspace(0.01, 0.06, 50)
        best_scale, best_ll = 0.03, 1e9
        for s in scales:
            p = sigmoid(s * X).clip(1e-6, 1 - 1e-6)
            ll = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
            if ll < best_ll:
                best_ll, best_scale = ll, s
        # Predict on validation
        Xv = df_va.get("elo_diff")
        if Xv is None:
            return np.full(len(df_va), 0.5, dtype=float)
        return sigmoid(best_scale * Xv.to_numpy()).clip(1e-6, 1 - 1e-6)
    else:
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

    return {
        "oof_path": str(save_dir / "oof_valid.csv"),
        "summary_path": str(save_dir / "backtest_results.csv"),
        "folds": len(val_year_windows),
    }
