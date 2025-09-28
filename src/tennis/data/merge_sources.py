from __future__ import annotations
import pandas as pd
import numpy as np
from .xref import _norm_name

def _player_id_to_name_map(player_xref: pd.DataFrame) -> pd.DataFrame:
    """
    Build a (player_id -> canonical_name) map using whatever columns exist.
    Preference order: name_sack > name > name_tml > name_tcb > name_norm
    (Only the ones present will be used.)
    """
    import pandas as pd

    m = player_xref.copy()

    # Ensure the expected identifier exists
    if "player_id" not in m.columns:
        return pd.DataFrame(columns=["player_id", "canonical_name"])

    # Collect any available name columns in priority order
    name_priority = ["name_sack", "name", "name_tml", "name_tcb", "name_norm"]
    available = [c for c in name_priority if c in m.columns and c != "player_id"]
    if not available:
        # Fall back to an empty string so downstream merges don’t crash
        m["canonical_name"] = pd.NA
    else:
        # First non-null across available columns
        m["canonical_name"] = (
            m[available]
            .astype("string")
            .bfill(axis=1)
            .iloc[:, 0]
            .astype("string")
        )

    out = (
        m[["player_id", "canonical_name"]]
        .dropna(subset=["player_id"])
        .drop_duplicates(subset=["player_id"])
    )
    return out


def _stable_match_key(df: pd.DataFrame) -> pd.Series:
    # hash on (tour, date, tourney_id/name, round, sorted players by canonical id if available)
    a = df["tour"].astype(str)
    b = df["date"].astype(str)
    c = df.get("tourney_id").astype(str).fillna("")
    d = _norm_name(df.get("tournament", df.get("tourney_name")))
    e = df.get("round").astype(str)
    p1 = df.get("winner_id").astype("Int64").fillna(-1).astype(str)
    p2 = df.get("loser_id").astype("Int64").fillna(-1).astype(str)
    pair = pd.DataFrame({"p1":p1,"p2":p2})
    # ensure order-invariant
    pair_sorted = pair.apply(lambda r: "|".join(sorted([r["p1"], r["p2"]])), axis=1)
    return (a + "|" + b + "|" + c + "|" + d + "|" + e + "|" + pair_sorted).map(hash)

def merge_to_curated(sack: pd.DataFrame, tml: pd.DataFrame, tcb: pd.DataFrame,
                     player_xref: pd.DataFrame, tourn_xref: pd.DataFrame):
    s = sack.copy()
    s["player_low"]  = s[["winner_id", "loser_id"]].min(axis=1)
    s["player_high"] = s[["winner_id", "loser_id"]].max(axis=1)
    s["player_a_id"] = s["player_low"]
    s["player_b_id"] = s["player_high"]
    s["y"] = (s["winner_id"] == s["player_a_id"]).astype(int)
    s = s.drop(columns=["player_low", "player_high"])
    s["match_key"] = _stable_match_key(s)

    sort_cols = [
        "date", "tour", "tournament", "round",
        "winner_id", "loser_id", "match_id_src"
    ]
    present_sort_cols = [c for c in sort_cols if c in s.columns]
    if present_sort_cols:
        s = s.sort_values(present_sort_cols)

    # Drop exact duplicates first (identical rows), then drop by key
    s = s.drop_duplicates()
    s = s.drop_duplicates(subset=["match_key"], keep="first")

    # Prepare TML & TCB with normalized join keys (name + date + tourney_name)
    def prep(df):
        if df.empty: 
            df = pd.DataFrame(columns=["date","tourney_name","winner_name","loser_name"])
        out = df.copy()
        id2name = _player_id_to_name_map(player_xref)

        if "winner_name" not in out.columns and "winner_id" in out.columns:
            out = out.merge(
                id2name.rename(columns={"player_id": "winner_id", "canonical_name": "winner_name"}),
                on="winner_id", how="left"
            )

        if "loser_name" not in out.columns and "loser_id" in out.columns:
            out = out.merge(
                id2name.rename(columns={"player_id": "loser_id", "canonical_name": "loser_name"}),
                on="loser_id", how="left"
            )

        # Dates & tournament normalization
        out["date"] = pd.to_datetime(out.get("date", out.get("tourney_date")))
        out["tourney_name_norm"] = _norm_name(out.get("tourney_name"))

        # Now safe to normalize player names (they exist or are empty strings)
        if "winner_name" not in out.columns:
            out["winner_name"] = ""   # fallback empty to keep length aligned
        if "loser_name" not in out.columns:
            out["loser_name"] = ""    # fallback empty to keep length aligned

        out["winner_name_norm"] = _norm_name(out["winner_name"])
        out["loser_name_norm"]  = _norm_name(out["loser_name"])

        return out

    tmlp = prep(tml)
    tcbp = prep(tcb)

    market_cols = [
        "open_odds_a", "open_odds_b",
        "close_odds_a", "close_odds_b",
        "odds_ts_open", "odds_ts_close",
    ]
    if tcbp is None or tcbp.empty:
        # if completely empty, create a shell with just the keys so left-join is harmless
        tcbp = pd.DataFrame(columns=[
            "date", "tourney_name", "winner_name", "loser_name",
            "tourney_name_norm", "winner_name_norm", "loser_name_norm", *market_cols
        ])
    else:
        # add any missing market columns as NA
        for c in market_cols:
            if c not in tcbp.columns:
                tcbp[c] = pd.NA

    # (optional but nice): cast odds to float
    for c in ["open_odds_a", "open_odds_b", "close_odds_a", "close_odds_b"]:
        if c in tcbp.columns:
            tcbp[c] = pd.to_numeric(tcbp[c], errors="coerce")

    # join TML by (date, tourney_name_norm, winner/loser name_norm)
    s["tourney_name_norm"] = _norm_name(s["tourney_name"])
    s["winner_name_norm"]  = _norm_name(s["winner_name"])
    s["loser_name_norm"]   = _norm_name(s["loser_name"])

    s_tml = s.merge(
        tmlp,
        on=["date","tourney_name_norm","winner_name_norm","loser_name_norm"],
        how="left",
        suffixes=("","_tml"),
    )

    has_markets = tcbp is not None and not tcbp.empty

    if has_markets:
        s_all = s_tml.merge(
            tcbp[[
                "date","tourney_name_norm","winner_name_norm","loser_name_norm",
                "open_odds_a","open_odds_b","close_odds_a","close_odds_b",
                "odds_ts_open","odds_ts_close",
            ]],
            on=["date","tourney_name_norm","winner_name_norm","loser_name_norm"],
            how="left"
        )
    else:
        # no markets available — just pass through
        s_all = s_tml.copy()
        for c in ["open_odds_a","open_odds_b","close_odds_a","close_odds_b","odds_ts_open","odds_ts_close"]:
            s_all[c] = pd.NA

    # Normalize stat column names (some sources use singular 'w_ace', others plural 'w_aces')
    rename_map = {
        "w_ace": "w_aces",
        "l_ace": "l_aces",
    }
    s_all = s_all.rename(columns={k: v for k, v in rename_map.items() if k in s_all.columns})

    # (optional) ensure all columns we’re about to select exist; if not, create NA
    needed_stats = [
        "match_key",
        "w_aces", "l_aces",
        "w_df", "l_df",
        "w_svpt", "l_svpt",
        "w_1stIn", "l_1stIn",
        "w_1stWon", "l_1stWon",
        "w_2ndWon", "l_2ndWon",
    ]
    for c in needed_stats:
        if c not in s_all.columns:
            s_all[c] = pd.NA

    # curated tables:
    cur_matches = (
        s[[
            "match_key","tour","date","tournament","level","surface","indoor","best_of",
            "winner_id","winner_name","loser_id","loser_name","score","retirement",
            "round","tourney_id","tourney_name","y"
        ]]
        .drop_duplicates()
        .drop_duplicates(subset=["match_key"], keep="first")
    )


    cur_stats = s_all[[
        "match_key","w_aces","l_aces","w_df","l_df","w_svpt","l_svpt","w_1stIn","l_1stIn","w_1stWon","l_1stWon","w_2ndWon","l_2ndWon"
    ]].drop_duplicates()

    cur_markets = s_all[[
        "match_key","open_odds_a","open_odds_b","close_odds_a","close_odds_b","odds_ts_open","odds_ts_close"
    ]].drop_duplicates()

    return cur_matches, cur_stats, cur_markets
