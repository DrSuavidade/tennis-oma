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

def add_boxscore_lagged_features(df: pd.DataFrame, window_matches: int = 10, min_hist: int = 3) -> pd.DataFrame:
    """
    Build per-player (A/B) serve/return *history* features using *past* matches only.
    We:
      1) derive A/B-side in-match rates,
      2) stack appearances for both players into a long table,
      3) for each player, compute shift().rolling(window) means,
      4) map the lagged means back to the original A/B rows,
      5) drop the current-match raw rates to avoid leakage.

    Required columns present (created as NaN if missing): 
      w_ace,l_ace,w_df,l_df,w_svpt,l_svpt,w_1stIn,l_1stIn,w_1stWon,l_1stWon,w_2ndWon,l_2ndWon
    Also needs: date, y (1 if player_a won), player_a_id, player_b_id
    """
    df = df.copy()
    # Ensure required numeric columns exist
    cols = [
        "w_ace","l_ace","w_df","l_df","w_svpt","l_svpt",
        "w_1stIn","l_1stIn","w_1stWon","l_1stWon","w_2ndWon","l_2ndWon"
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    # Assign winner/loser stats to A/B sides for THIS match (intermediate; will be lagged)
    def to_ab(w_col, l_col):
        a = np.where(df["y"] == 1, df[w_col], df[l_col])
        b = np.where(df["y"] == 1, df[l_col], df[w_col])
        return pd.Series(a, index=df.index), pd.Series(b, index=df.index)

    a_svpt, b_svpt = to_ab("w_svpt", "l_svpt")
    a_ace,  b_ace  = to_ab("w_ace",  "l_ace")
    a_df,   b_df   = to_ab("w_df",   "l_df")
    a_1in,  b_1in  = to_ab("w_1stIn","l_1stIn")
    a_1won, b_1won = to_ab("w_1stWon","l_1stWon")
    a_2won, b_2won = to_ab("w_2ndWon","l_2ndWon")

    # Safe divide
    def rate(numer, denom):
        numer = pd.to_numeric(numer, errors="coerce")
        denom = pd.to_numeric(denom, errors="coerce")
        out = numer / denom
        return out.replace([np.inf, -np.inf], np.nan)

    # Current-match rates (we will ONLY use their lagged/rolling versions)
    cur = pd.DataFrame(index=df.index)
    cur["a_spw"]        = rate(a_1won + a_2won, a_svpt)
    cur["b_spw"]        = rate(b_1won + b_2won, b_svpt)
    cur["a_ace_rate"]   = rate(a_ace, a_svpt)
    cur["b_ace_rate"]   = rate(b_ace, b_svpt)
    cur["a_df_rate"]    = rate(a_df,  a_svpt)
    cur["b_df_rate"]    = rate(b_df,  b_svpt)
    cur["a_first_in"]   = rate(a_1in, a_svpt)
    cur["b_first_in"]   = rate(b_1in, b_svpt)
    cur["a_first_won"]  = rate(a_1won, a_1in)
    cur["b_first_won"]  = rate(b_1won, b_1in)
    cur["a_second_won"] = rate(a_2won, a_svpt - a_1in)
    cur["b_second_won"] = rate(b_2won, b_svpt - b_1in)

    # Build a long "player appearances" table to compute per-player history
    base_cols = ["spw","ace_rate","df_rate","first_in","first_won","second_won"]
    a_long = pd.DataFrame({
        "_row": df.index,
        "side": "A",
        "player_id": df["player_a_id"],
        "date": pd.to_datetime(df["date"]),
        "spw": cur["a_spw"],
        "ace_rate": cur["a_ace_rate"],
        "df_rate": cur["a_df_rate"],
        "first_in": cur["a_first_in"],
        "first_won": cur["a_first_won"],
        "second_won": cur["a_second_won"],
    })
    b_long = pd.DataFrame({
        "_row": df.index,
        "side": "B",
        "player_id": df["player_b_id"],
        "date": pd.to_datetime(df["date"]),
        "spw": cur["b_spw"],
        "ace_rate": cur["b_ace_rate"],
        "df_rate": cur["b_df_rate"],
        "first_in": cur["b_first_in"],
        "first_won": cur["b_first_won"],
        "second_won": cur["b_second_won"],
    })

    long = pd.concat([a_long, b_long], ignore_index=True)
    long = long.sort_values(["player_id", "date", "_row"])

    # For each metric, compute lagged rolling means over a player's past matches
    for m in base_cols:
        long[f"{m}_m{window_matches}"] = (
            long.groupby("player_id")[m]
                .apply(lambda s: s.shift(1).rolling(window_matches, min_periods=min_hist).mean())
                .reset_index(level=0, drop=True)
        )

    # Pull the lagged features back to the wide (A/B) frame
    a_hist = (long[long["side"]=="A"]
                .set_index("_row")[[f"{m}_m{window_matches}" for m in base_cols]]
                .rename(columns=lambda c: "a_" + c))
    b_hist = (long[long["side"]=="B"]
                .set_index("_row")[[f"{m}_m{window_matches}" for m in base_cols]]
                .rename(columns=lambda c: "b_" + c))

    df = df.join(a_hist, how="left").join(b_hist, how="left")

    # Fill missing histories with 0 (player with no prior matches)
    hist_cols = [f"{p}_{m}_m{window_matches}" for p in ["a","b"] for m in base_cols]
    df[hist_cols] = df[hist_cols].fillna(0.0)

    # Do NOT keep current-match raw rates to avoid leakage
    # (they were only an intermediate to build history)
    # If any of these columns exist from previous steps, make sure they’re gone:
    to_drop = [
        "a_spw","b_spw","a_ace_rate","b_ace_rate","a_df_rate","b_df_rate",
        "a_first_in","b_first_in","a_first_won","b_first_won","a_second_won","b_second_won"
    ]
    existing = [c for c in to_drop if c in df.columns]
    if existing:
        df = df.drop(columns=existing)

    return df


def finalize_features(df: pd.DataFrame) -> pd.DataFrame:
    # --- Elo helpers (ADD THIS BLOCK) ---
    # elo_diff: surface-adjusted Elo difference (A - B), fallback from elo_exp_a if needed
    if "elo_a_surface" in df.columns and "elo_b_surface" in df.columns:
        df["elo_diff"] = (df["elo_a_surface"] - df["elo_b_surface"]).astype(float)
    else:
        # recover a pseudo-diff from prob if only prob exists
        if "elo_exp_a" in df.columns:
            p_clip = df["elo_exp_a"].clip(1e-6, 1-1e-6).astype(float)
            df["elo_diff"] = np.log(p_clip / (1.0 - p_clip))
        else:
            df["elo_diff"] = 0.0

    # optional: keep a logit of the Elo win prob for tree models
    if "elo_exp_a" in df.columns:
        p_clip = df["elo_exp_a"].clip(1e-6, 1-1e-6).astype(float)
        df["elo_logit"] = np.log(p_clip / (1.0 - p_clip))
    else:
        df["elo_logit"] = 0.0
    # --- end Elo helpers ---
    feat_cols = [
        # ratings
        "elo_a_surface", "elo_b_surface", "elo_exp_a",
        # form
        "a_matches_30d", "b_matches_30d",
        # box score history (10 last matches)
        "a_spw_m10","b_spw_m10",
        "a_ace_rate_m10","b_ace_rate_m10",
        "a_df_rate_m10","b_df_rate_m10",
        "a_first_in_m10","b_first_in_m10",
        "a_first_won_m10","b_first_won_m10",
        "a_second_won_m10","b_second_won_m10",
        # context
        "indoor", "best_of",
    ]
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0.0
    if "y" in df.columns:
        df["target"] = df["y"].astype(int)
    cols = ["match_id", "date", "tour", "surface", "player_a_id", "player_b_id", *feat_cols, "target"]
    return df[cols]

def make_symmetric(feats: pd.DataFrame) -> pd.DataFrame:
    """
    Return a balanced dataset by adding a swapped-row copy:
      - swap *_a <-> *_b columns
      - flip y := 1 - y
      - recompute elo_exp_a as the opposite side's probability (1 - elo_exp_a)
    Assumes the finalized feature columns from finalize_features().
    """
    feats = feats.copy()

    # Columns to swap (adjust if you add/remove features later)
    swap_pairs = [
        ("elo_a_surface", "elo_b_surface"),
        ("a_matches_30d", "b_matches_30d"),
    ]

    # Build swapped copy
    swapped = feats.copy()
    for a, b in swap_pairs:
        if a in swapped.columns and b in swapped.columns:
            tmp = swapped[a].copy()
            swapped[a] = swapped[b]
            swapped[b] = tmp

    # elo_exp_a is the probability from A's perspective.
    # When we swap players, A becomes previous B, so P(A wins) = 1 - old P(A wins).
    if "elo_exp_a" in swapped.columns:
        swapped["elo_exp_a"] = 1.0 - swapped["elo_exp_a"]

    # Flip target
    if "y" in swapped.columns:
        swapped["y"] = 1 - swapped["y"]

    # Optional: give the swapped rows a different id if you carry one around
    # (not required for training/stacking)
    # if "match_key" in swapped.columns:
    #     swapped["match_key"] = swapped["match_key"].astype(str) + "_r"

    # Combine original + swapped
    out = pd.concat([feats, swapped], ignore_index=True)

    # (Optional) sanity: drop exact duplicates, if any
    out = out.drop_duplicates().reset_index(drop=True)
    return out



