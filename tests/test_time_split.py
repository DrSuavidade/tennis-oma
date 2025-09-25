import pandas as pd
from tennis.utils.time_split import rolling_time_splits

def test_splits():
    dates = pd.date_range("2017-01-01", periods=200, freq="D")
    df = pd.DataFrame({"date": dates})
    folds = rolling_time_splits(df, "date", "W", n_folds=3, train_min_weeks=52, valid_weeks=4, test_weeks=4)
    assert len(folds) >= 1
