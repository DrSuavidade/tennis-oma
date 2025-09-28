# src/tennis/data/xref.py
from __future__ import annotations
import pandas as pd
import re

def _norm_name(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype=str)
    s = s.fillna("").astype(str)
    s = s.str.lower()
    s = s.str.replace(r"[^a-z\s\-']", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s

def _ensure_name_columns(df: pd.DataFrame, s_players: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has 'winner_name' and 'loser_name'.
    If absent, try common alternatives; if still missing, map from winner_id/loser_id via s_players.
    """
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()

    out = df.copy()
    cols = {c.lower(): c for c in out.columns}

    # Try direct alternative name columns first
    def try_set(target, alts):
        if target not in out.columns:
            for a in alts:
                if a.lower() in cols:
                    out[target] = out[cols[a.lower()]]
                    return True
        return target in out.columns

    _ = try_set("winner_name", ["winner", "w_name", "winner_player_name", "wplayer"])
    _ = try_set("loser_name",  ["loser",  "l_name", "loser_player_name", "lplayer"])

    # Map from IDs if still missing
    if "winner_name" not in out.columns and "winner_id" in cols:
        tmp = out.merge(
            s_players[["player_id", "name"]].rename(columns={"name": "winner_name"}),
            left_on=cols["winner_id"], right_on="player_id", how="left"
        )
        out["winner_name"] = tmp["winner_name"].values

    if "loser_name" not in out.columns and "loser_id" in cols:
        tmp = out.merge(
            s_players[["player_id", "name"]].rename(columns={"name": "loser_name"}),
            left_on=cols["loser_id"], right_on="player_id", how="left"
        )
        out["loser_name"] = tmp["loser_name"].values

    return out

def uniq_names(df: pd.DataFrame, wcol: str, lcol: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["name", "name_norm"]).drop_duplicates()
    n1 = df[[wcol]].rename(columns={wcol: "name"})
    n2 = df[[lcol]].rename(columns={lcol: "name"})
    names = pd.concat([n1, n2], ignore_index=True).dropna().drop_duplicates()
    names["name_norm"] = _norm_name(names["name"])
    return names

def build_player_xref(sack: pd.DataFrame, tml: pd.DataFrame, tcb: pd.DataFrame) -> pd.DataFrame:
    # Sackmann canonical player list
    # winner_name / loser_name are always present here
    s_names = pd.concat(
        [
            sack[["winner_id", "winner_name"]].rename(columns={"winner_id": "player_id", "winner_name": "name"}),
            sack[["loser_id",  "loser_name"]].rename(columns={"loser_id":  "player_id", "loser_name":  "name"}),
        ],
        ignore_index=True,
    ).dropna().drop_duplicates()

    # Some rows (esp. Davis Cup) may not have IDs; keep name-only uniques as well
    s_names = s_names.drop_duplicates(subset=["player_id", "name"])
    s_players = s_names.dropna(subset=["player_id"]).drop_duplicates("player_id")
    s_players["name_norm"] = _norm_name(s_players["name"])

    # Ensure TML/TCB have names; if not, infer from IDs using Sackmann mapping
    tml = _ensure_name_columns(tml, s_players)
    tcb = _ensure_name_columns(tcb, s_players)

    # Name tables
    tml_names = uniq_names(tml, "winner_name", "loser_name")
    tcb_names = uniq_names(tcb, "winner_name", "loser_name")

    # Left-join: name_norm -> Sackmann canonical (player_id, name)
    # Prefer exact normalized match; duplicates are fine, we dedupe later.
    joined = []
    for src in [tml_names, tcb_names]:
        if not src.empty:
            j = src.merge(
                s_players[["player_id", "name", "name_norm"]],
                on="name_norm", how="left", suffixes=("", "_sack")
            )
            joined.append(j)

    if joined:
        out = pd.concat(joined, ignore_index=True)
    else:
        # If both TML + TCB are empty, fall back to Sackmann names only
        out = s_players[["player_id", "name", "name_norm"]].copy()

    # Deduplicate: prefer rows that actually resolved to a player_id
    out["has_id"] = out["player_id"].notna()
    out = out.sort_values(["has_id", "name_norm"], ascending=[False, True])
    out = out.drop_duplicates(subset=["name_norm"], keep="first").drop(columns=["has_id"], errors="ignore")

    return out.reset_index(drop=True)

def build_tournament_xref(sack: pd.DataFrame, tml: pd.DataFrame, tcb: pd.DataFrame) -> pd.DataFrame:
    # Existing logic can stay; if it expects names/IDs, add similar guards as above if needed.
    s_tour = sack[["tourney_id", "tourney_name"]].drop_duplicates()
    t_tour = tml[["tourney_id", "tourney_name"]].drop_duplicates() if tml is not None and not tml.empty else pd.DataFrame(columns=["tourney_id","tourney_name"])
    frames = []
    for df in (s_tour, t_tour):
        if df is not None and not df.empty:
            # remove columns that are entirely NA to avoid future concat dtype changes
            df = df.loc[:, df.notna().any(axis=0)]
            if not df.empty and df.shape[1] > 0:
                frames.append(df)

    if frames:
        x = pd.concat(frames, ignore_index=True)
    else:
        # empty scaffold with expected columns (adjust if your function expects others)
        x = pd.DataFrame(columns=["tourney_id", "tourney_name"])

    # Drop rows where *all* key fields are NA; then de-dupe
    key_cols = [c for c in x.columns if c.endswith("id") or c.endswith("name")]
    if key_cols:
        x = x.dropna(subset=key_cols, how="all")
    x = x.drop_duplicates()

    x["tourney_norm"] = _norm_name(x["tourney_name"])
    return x
