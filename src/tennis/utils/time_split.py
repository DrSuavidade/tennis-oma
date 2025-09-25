import pandas as pd
from typing import List, Tuple

def rolling_time_splits(df: pd.DataFrame, date_col: str, freq: str, n_folds: int,
                        train_min_weeks: int, valid_weeks: int, test_weeks: int):
    df = df.sort_values(date_col)
    start = df[date_col].min().normalize()
    end = df[date_col].max().normalize()
    cutpoints = pd.date_range(start=start, end=end, freq=freq)

    folds = []
    for i in range(n_folds):
        # train end -> valid start
        valid_start = cutpoints[min(i + train_min_weeks, len(cutpoints)-1)]
        valid_end = valid_start + pd.Timedelta(weeks=valid_weeks)
        test_end  = valid_end + pd.Timedelta(weeks=test_weeks)

        train_idx = (df[date_col] < valid_start)
        valid_idx = (df[date_col] >= valid_start) & (df[date_col] < valid_end)
        test_idx  = (df[date_col] >= valid_end) & (df[date_col] < test_end)

        if valid_idx.sum() == 0 or test_idx.sum() == 0:
            break
        folds.append((train_idx, valid_idx, test_idx, valid_start, valid_end))

    return folds
