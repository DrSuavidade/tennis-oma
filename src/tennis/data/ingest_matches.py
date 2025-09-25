import pandas as pd
from pathlib import Path

def load_synth(path: str) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_csv(p / "synth_matches.csv", parse_dates=["date"])
    return df

def load_from_sackmann(atp_dir: str, wta_dir: str) -> pd.DataFrame:
    # Placeholder: user can implement reading yearly CSVs and concatenating
    # while normalizing columns to the schema.
    raise NotImplementedError("Implement parsing of Sackmann CSVs here.")

def load(path_or_source: str, tour: str = "both") -> pd.DataFrame:
    if path_or_source == "synth":
        df = load_synth("data/raw/synth")
    else:
        raise NotImplementedError("Extend load() for real sources.")

    if tour in ("ATP", "WTA"):
        df = df[df["tour"] == tour]
    return df.reset_index(drop=True)
