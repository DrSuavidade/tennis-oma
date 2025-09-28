from pathlib import Path
import pandas as pd

def load_tcb_markets(tcb_dir: str, years=None) -> pd.DataFrame:
    """
    Returns per-match market lines (opening/closing).
    Keep columns:
      - winner_name, loser_name (or playerA/B if provided)
      - tourney_name, round, surface, date (or timestamp)
      - close_odds_a, close_odds_b, open_odds_a, open_odds_b
      - odds_ts_close, odds_ts_open (if available)
    """
    root = Path(tcb_dir)
    files = sorted(root.rglob("*.csv"))
    frames = []
    for f in files:
        df = pd.read_csv(f)
        cols = {c.lower(): c for c in df.columns}
        out = pd.DataFrame()
        out["tourney_name"] = df.get(cols.get("tourney_name"), df.get(cols.get("tournament")))
        out["date"] = pd.to_datetime(df.get(cols.get("date")), errors="coerce")
        out["round"] = df.get(cols.get("round"))
        # try both naming variants
        out["winner_name"] = df.get(cols.get("winner_name"), df.get(cols.get("player1")))
        out["loser_name"] = df.get(cols.get("loser_name"), df.get(cols.get("player2")))
        out["close_odds_a"] = df.get(cols.get("close_odds_a"), df.get(cols.get("odds1_close")))
        out["close_odds_b"] = df.get(cols.get("close_odds_b"), df.get(cols.get("odds2_close")))
        out["open_odds_a"] = df.get(cols.get("open_odds_a"), df.get(cols.get("odds1_open")))
        out["open_odds_b"] = df.get(cols.get("open_odds_b"), df.get(cols.get("odds2_open")))
        out["odds_ts_close"] = pd.to_datetime(df.get(cols.get("odds_ts_close")), errors="coerce")
        out["odds_ts_open"] = pd.to_datetime(df.get(cols.get("odds_ts_open")), errors="coerce")
        out["match_id_src"] = f.name
        frames.append(out)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
