# src/tennis/data/ingest_tml.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
from typing import Iterable

def _read_csv_safe(path: Path) -> pd.DataFrame:
    """Try UTF-8 first, then latin-1. Never crash on encoding."""
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")

def _find_year_files(root: Path, years: Iterable[int]) -> list[Path]:
    """
    Only pick files named exactly YYYY.csv in the root or any subdir.
    Skip player bios like ATP_Database.csv, and other non-match tables.
    """
    out = []
    years = list(years)
    for y in years:
        # common locations: root/2013.csv or root/some_subdir/2013.csv
        candidates = list(root.glob(f"{y}.csv")) + list(root.glob(f"**/{y}.csv"))
        out.extend(c for c in candidates if c.is_file())
    # de-duplicate while preserving order
    seen = set()
    unique = []
    for p in out:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique

def load_tml_stats(tml_dir: str, years: Iterable[int] | range) -> pd.DataFrame:
    """
    Loads *per-match* stat tables from the TML-Database (year CSVs only).
    Returns a standardized dataframe with a subset of columns we will merge on.
    """
    root = Path(tml_dir)
    files = _find_year_files(root, years)

    frames: list[pd.DataFrame] = []
    for f in files:
        df = _read_csv_safe(f)

        # Column map helper (case-insensitive)
        cols = {c.lower(): c for c in df.columns}

        def get(col: str, default=None):
            return df[cols[col]] if col in cols else default

        # Robust tourney_date parsing (TML should be ISO or YYYYMMDD; handle both)
        td = get("tourney_date")
        if td is not None:
            if pd.api.types.is_integer_dtype(td) or pd.api.types.is_float_dtype(td):
                # if it looks like YYYYMMDD integers
                # NOTE: floats might be because of CSV read; cast to Int64 safely
                td2 = pd.to_numeric(td, errors="coerce").astype("Int64")
                date = pd.to_datetime(td2, format="%Y%m%d", errors="coerce")
            else:
                date = pd.to_datetime(td, errors="coerce")
        else:
            date = pd.NaT

        out = pd.DataFrame({
            "tourney_id": get("tourney_id"),
            "tourney_name": get("tourney_name"),
            "surface": get("surface"),
            "round": get("round"),
            "tourney_level": get("tourney_level"),
            "date": date,
            "match_num": get("match_num"),
            "winner_id": pd.to_numeric(get("winner_id"), errors="coerce").astype("Int64"),
            "loser_id": pd.to_numeric(get("loser_id"), errors="coerce").astype("Int64"),
            # A small subset of serve/return stats we actually use; add more later
            "w_ace": get("w_ace"),
            "w_df": get("w_df"),
            "w_svpt": get("w_svpt"),
            "w_1stIn": get("w_1stIn"),
            "w_1stWon": get("w_1stWon"),
            "w_2ndWon": get("w_2ndWon"),
            "w_bpSaved": get("w_bpSaved"),
            "w_bpFaced": get("w_bpFaced"),
            "l_ace": get("l_ace"),
            "l_df": get("l_df"),
            "l_svpt": get("l_svpt"),
            "l_1stIn": get("l_1stIn"),
            "l_1stWon": get("l_1stWon"),
            "l_2ndWon": get("l_2ndWon"),
            "l_bpSaved": get("l_bpSaved"),
            "l_bpFaced": get("l_bpFaced"),
        })
        out["match_id_src"] = f.name
        frames.append(out)

    if not frames:
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)

    # Keep only requested year range (based on the parsed date)
    df_all = df_all[df_all["date"].notna()]
    yrs = set(int(y) for y in years)
    df_all = df_all[df_all["date"].dt.year.isin(yrs)]

    return df_all.reset_index(drop=True)
