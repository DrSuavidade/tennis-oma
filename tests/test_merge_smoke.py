from pathlib import Path
import pandas as pd

def test_curated_exist():
    # Only checks presence if freeze was run
    asof_dirs = list(Path("data/processed").glob("asof=*"))
    if not asof_dirs:
        return
    base = asof_dirs[0]
    assert (base / "cur_matches.parquet").exists()
