from pathlib import Path
import pandas as pd

def _read_years(root: Path, pattern: str, years=None) -> list[Path]:
    # Collect files, filter to requested years, exclude doubles/qualifying
    files = sorted(root.glob(pattern))
    if years is not None:
        yrs = set(map(int, years)) if not isinstance(years, range) else set(years)
        files = [p for p in files if any(str(y) in p.name for y in yrs)]
    # singles only
    files = [p for p in files if "doubles" not in p.name.lower() and "qual" not in p.name.lower()]
    return files

def _series(df: pd.DataFrame, col: str, default) -> pd.Series:
    # Safe column getter: returns a Series aligned to df.index
    if col in df.columns:
        return df[col]
    return pd.Series(default, index=df.index)

def load_sackmann_matches(atp_dir: str, wta_dir: str, years=None) -> pd.DataFrame:
    """
    Returns unified long table for ATP+WTA with canonical columns:
    ['tour','date','tournament','level','surface','indoor','best_of',
     'winner_id','winner_name','loser_id','loser_name','score','retirement',
     'round','tourney_id','tourney_name','match_id_src']
    """
    frames = []
    for tour, root in [("ATP", Path(atp_dir)), ("WTA", Path(wta_dir))]:
        files = _read_years(root, f"{tour.lower()}_matches_*.csv", years)
        for f in files:
            df = pd.read_csv(f)

            # Normalize essentials
            df["tour"] = tour
            # Sackmann has 'tourney_date' as yyyymmdd int; some years include 'date'
            date_col = "tourney_date" if "tourney_date" in df.columns else "date"
            df["date"] = pd.to_datetime(df[date_col], format="%Y%m%d", errors="coerce") \
                          if date_col == "tourney_date" else pd.to_datetime(df[date_col], errors="coerce")

            # Safe optional fields
            indoor = _series(df, "indoor", 0)
            best_of = _series(df, "best_of", 3)
            retirement = _series(df, "retirement", 0)

            # Winner/loser IDs (some historical files may differ slightly)
            w_id = _series(df, "winner_id", _series(df, "winner_player_id", pd.NA)).astype("Int64")
            l_id = _series(df, "loser_id",  _series(df, "loser_player_id",  pd.NA)).astype("Int64")

            out = pd.DataFrame({
                "tour": df["tour"],
                "date": df["date"],
                "tournament": _series(df, "tourney_name", _series(df, "tournament", "")),
                "level": _series(df, "tourney_level", _series(df, "level", pd.NA)),
                "surface": _series(df, "surface", pd.NA),
                "indoor": indoor.fillna(0).astype(int),
                "best_of": best_of.fillna(3).astype(int),
                "winner_id": w_id,
                "winner_name": _series(df, "winner_name", ""),
                "loser_id": l_id,
                "loser_name": _series(df, "loser_name", ""),
                "score": _series(df, "score", ""),
                "retirement": retirement.fillna(0).astype(int),
                "round": _series(df, "round", pd.NA),
                "tourney_id": _series(df, "tourney_id", pd.NA),
                "tourney_name": _series(df, "tourney_name", _series(df, "tournament", "")),
            })
            out["match_id_src"] = f.name  # provenance
            frames.append(out)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
