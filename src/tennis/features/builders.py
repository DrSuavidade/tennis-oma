import pandas as pd
import numpy as np

ROLL_WINDOWS = [5, 10, 20]

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "y" not in df.columns:
        # fallback dummy: derive from score if present
        df["y"] = (df["player_a_sets"] >= df["player_b_sets"]).astype(int) if "player_a_sets" in df.columns else 0
    df["y"] = df["y"].astype(int)
    return df

def add_rolling_form(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    def make_30d_count(frame: pd.DataFrame, pid_col: str, out_col: str) -> pd.Series:
        # For each player, count matches in the past 30 days (inclusive of current day)
        tmp = (
            frame
            .sort_values([pid_col, "date"])
            .set_index("date")
            .groupby(pid_col)["match_id"]
            .rolling("30D")
            .count()
            .reset_index()              # columns: [pid_col, 'date', 'match_id']
            .rename(columns={"match_id": out_col})
        )
        # Merge back to original rows on (pid, date)
        out = frame.merge(tmp, on=[pid_col, "date"], how="left")[out_col]
        return out

    df["a_matches_30d"] = make_30d_count(df, "player_a_id", "a_matches_30d")
    df["b_matches_30d"] = make_30d_count(df, "player_b_id", "b_matches_30d")

    df[["a_matches_30d", "b_matches_30d"]] = df[["a_matches_30d", "b_matches_30d"]].fillna(0.0)
    return df


def finalize_features(df: pd.DataFrame) -> pd.DataFrame:
    feat_cols = [
        "elo_a_surface", "elo_b_surface", "elo_exp_a",
        "a_matches_30d", "b_matches_30d",
        "indoor", "best_of",
    ]
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0.0
    df["target"] = df["y"].astype(int)
    cols = ["match_id", "date", "tour", "surface", "player_a_id", "player_b_id", *feat_cols, "target"]
    return df[cols]
