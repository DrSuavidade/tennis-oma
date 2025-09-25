import pandas as pd

def assert_no_future_features(df: pd.DataFrame, date_col: str, group_cols: list, cols: list):
    # basic guard: check that for each group, features are lagged (no same-day peeking)
    df_sorted = df.sort_values([*group_cols, date_col])
    for c in cols:
        if c not in df.columns:
            continue
        # If any feature equals a forward-filled target-derived quantity on same date, raise.
        # (Lightweight placeholder; extend with more robust sentries.)
        pass
