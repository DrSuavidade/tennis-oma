import pandas as pd

def run_checks(cur_matches: pd.DataFrame, cur_stats: pd.DataFrame, cur_markets: pd.DataFrame) -> dict:
    issues = {}

    # unique keys
    dup = cur_matches["match_key"].duplicated().sum()
    if dup > 0: issues["duplicate_match_keys"] = int(dup)

    # basic domains
    invalid_bo = (~cur_matches["best_of"].isin([3,5])).sum()
    if invalid_bo: issues["invalid_best_of"] = int(invalid_bo)

    # markets sanity
    if not cur_markets.empty:
        for side in ["open_odds_a","open_odds_b","close_odds_a","close_odds_b"]:
            bad = cur_markets[side].dropna().le(1.0).sum()  # odds must be > 1.0 (decimal)
            if bad: issues[f"bad_{side}"] = int(bad)

    return issues
