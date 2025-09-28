# src/tennis/data/normalize.py
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Iterable, List

# --------- CONFIG ----------
# Columns we guarantee in staging
STAGING_SCHEMA = [
    "tour", "date", "tournament", "level", "surface", "indoor", "best_of",
    "winner_id", "winner_name", "loser_id", "loser_name",
    "score", "retirement", "round",
    "tourney_id", "tourney_name",
    "match_id_src",
    # keep raw stat cols when present (Sackmann naming)
    "w_ace","w_df","w_svpt","w_1stIn","w_1stWon","w_2ndWon","w_bpSaved","w_bpFaced",
    "l_ace","l_df","l_svpt","l_1stIn","l_1stWon","l_2ndWon","l_bpSaved","l_bpFaced",
]

# "Significant" fields required for ML later
REQUIRED_NON_NULL = [
    "tour", "date", "winner_id", "loser_id", "tourney_name", "round", "best_of", "surface"
]

# Regexes we treat as retirement/walkover defaults
RET_TOKENS = ("RET", "W/O", "DEF", "ABN")

# --------- HELPERS ----------
def _safe_parse_date(series: pd.Series) -> pd.Series:
    """
    Accepts either integer yyyymmdd (Sackmann tourney_date) or ISO date strings.
    """
    s = series.copy()
    # int yyyymmdd -> string
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
        s = s.fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
    return pd.to_datetime(s, errors="coerce", format=None)

def _norm_surface(s: pd.Series) -> pd.Series:
    s = s.fillna("").str.strip().str.title()
    # Normalize common variants
    m = {
        "Carpet":"Carpet", "Hard":"Hard", "Clay":"Clay", "Grass":"Grass",
        "":"Unknown", "Acrylic":"Hard", "Cement":"Hard", "Greenset":"Hard",
    }
    return s.map(lambda x: m.get(x, x))

def _detect_retirement(score: pd.Series) -> pd.Series:
    s = score.fillna("").astype(str).str.upper()
    return s.apply(lambda x: any(tok in x for tok in RET_TOKENS))

def _ensure_all_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for c in columns:
        if c not in df.columns:
            df[c] = pd.NA
    return df

def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    # types that downstream expects
    int_like = ["winner_id","loser_id","best_of"]
    for c in int_like:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    if "indoor" in df.columns:
        df["indoor"] = pd.to_numeric(df["indoor"], errors="coerce").fillna(0).astype(int)
    # stats numeric
    string_cols = [
        "level", "round", "tournament", "tourney_name", "tourney_id",
        "score", "winner_name", "loser_name", "match_id_src", "surface", "tour"
    ]
    for c in string_cols:
        if c in df.columns:
            df[c] = df[c].astype("string")
    return df

# --------- NORMALIZERS ----------
def normalize_sackmann_match_df(raw: pd.DataFrame, tour_label: str, src_name: str) -> pd.DataFrame:
    """
    Normalize one Sackmann match CSV (ATP or WTA) to the staging schema.
    Keeps singles only (we'll exclude doubles files by file selection).
    """
    df = raw.copy()

    # prefer 'tourney_date' else 'date'
    if "tourney_date" in df.columns:
        df["date"] = _safe_parse_date(df["tourney_date"])
    elif "date" in df.columns:
        df["date"] = _safe_parse_date(df["date"])
    else:
        df["date"] = pd.NaT

    # tournament naming
    df["tournament"] = df.get("tourney_name", df.get("tournament", pd.NA))
    df["tourney_name"] = df.get("tourney_name", df["tournament"])
    df["tourney_id"] = df.get("tourney_id")
    df["level"] = df.get("tourney_level", df.get("level", pd.NA))
    df["surface"] = _norm_surface(df.get("surface", pd.NA))
    df["best_of"] = df.get("best_of", pd.NA)
    df["round"] = df.get("round", pd.NA)
    df["winner_id"] = df.get("winner_id", pd.NA)
    df["winner_name"] = df.get("winner_name", pd.NA)
    df["loser_id"] = df.get("loser_id", pd.NA)
    df["loser_name"] = df.get("loser_name", pd.NA)
    df["score"] = df.get("score", pd.NA)
    df["retirement"] = _detect_retirement(df["score"])
    df["indoor"] = df.get("indoor", 0)  # Sackmann doesn’t include indoor; keep 0
    df["tour"] = tour_label
    df["match_id_src"] = src_name

    # stats columns (if present)
    stat_cols = ["w_ace","w_df","w_svpt","w_1stIn","w_1stWon","w_2ndWon","w_bpSaved","w_bpFaced",
                 "l_ace","l_df","l_svpt","l_1stIn","l_1stWon","l_2ndWon","l_bpSaved","l_bpFaced"]
    for c in stat_cols:
        if c not in df.columns:
            df[c] = pd.NA

    df = _ensure_all_columns(df, STAGING_SCHEMA)
    df = _coerce_types(df)

    # basic quality filter
    # NOTE: keep rows even if stats are NaN; but drop if key metadata missing
    mask = pd.Series(True, index=df.index)
    for c in REQUIRED_NON_NULL:
        mask &= df[c].notna()
    # also drop dates that failed to parse
    mask &= df["date"].notna()

    df = df.loc[mask, STAGING_SCHEMA].drop_duplicates()
    return df

def read_all_sackmann(
    atp_dir: Path,
    wta_dir: Path,
    include_quals_chall: bool = True,
    include_futures_itf: bool = True,
) -> pd.DataFrame:
    """
    Read & normalize all relevant Sackmann files (singles only) into one DataFrame.
    """
    def _glob(dirpath: Path, patterns: Iterable[str]) -> List[Path]:
        out = []
        for pat in patterns:
            out.extend(sorted(dirpath.glob(pat)))
        return out

    # ATP singles main + optional ladders
    atp_files = _glob(atp_dir, ["atp_matches_*.csv"])
    # exclude known doubles CSV by name (we won't glob 'doubles')
    atp_exclude = {"atp_matches_doubles", "atp_matches_quali_doubles"}
    atp_files = [p for p in atp_files if not any(ex in p.name for ex in atp_exclude)]

    if include_quals_chall:
        atp_files += _glob(atp_dir, ["atp_matches_qual_chall_*.csv"])
    if include_futures_itf:
        atp_files += _glob(atp_dir, ["atp_matches_futures_*.csv"])

    # WTA singles main + optional ITF qualifiers
    wta_files = _glob(wta_dir, ["wta_matches_*.csv"])
    # keep qual/itf as well when asked
    if include_futures_itf:
        wta_files += _glob(wta_dir, ["wta_matches_qual_itf_*.csv"])

    frames = []

    for p in atp_files:
        try:
            raw = pd.read_csv(p, low_memory=False)
            frames.append(normalize_sackmann_match_df(raw, "ATP", p.name))
        except Exception as e:
            print(f"[normalize] Skipping {p.name}: {e}")

    for p in wta_files:
        try:
            raw = pd.read_csv(p, low_memory=False)
            frames.append(normalize_sackmann_match_df(raw, "WTA", p.name))
        except Exception as e:
            print(f"[normalize] Skipping {p.name}: {e}")

    if not frames:
        return pd.DataFrame(columns=STAGING_SCHEMA)

    out = pd.concat(frames, ignore_index=True)
    # Global de-dup by (tour, date, tourney_name, winner_id, loser_id, score)
    out = out.sort_values("date").drop_duplicates(
        subset=["tour","date","tourney_name","winner_id","loser_id","score"], keep="first"
    )
    return out

def write_staging(matches: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # main staging file (replacing old stg_tml_stats.stg_tcb_markets expectations)
    (out_dir / "stg_matches.parquet").write_bytes(
        matches.to_parquet(index=False)
    )

def build_staging_from_external(
    external_root: Path,
    out_dir: Path,
    include_quals_chall: bool = True,
    include_futures_itf: bool = True,
) -> pd.DataFrame:
    """
    Public entrypoint used by CLI: reads all external repos and writes normalized staging.
    """
    atp_dir = external_root / "tennis_atp"
    wta_dir = external_root / "tennis_wta"

    matches = read_all_sackmann(
        atp_dir=atp_dir,
        wta_dir=wta_dir,
        include_quals_chall=include_quals_chall,
        include_futures_itf=include_futures_itf,
    )

    write_staging(matches, out_dir)
    return matches
