import pandas as pd
import numpy as np
import lightgbm as lgb

FEATURES = [
    "elo_a_surface", "elo_b_surface", "elo_exp_a",
    "a_matches_30d", "b_matches_30d",
    "indoor", "best_of",
]


class LGBMModel:
    def __init__(self, max_rounds=300, early_stopping_rounds=30, learning_rate=0.05,
                 num_leaves=64, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1):
        self.params = dict(
            objective="binary",
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            feature_fraction=feature_fraction,
            bagging_fraction=bagging_fraction,
            bagging_freq=bagging_freq,
            metric="binary_logloss",
            verbose=-1,
            force_row_wise=True,
            seed=42,
            feature_fraction_seed=42,
            bagging_seed=42,
            deterministic=True,
        )

        self.max_rounds = max_rounds
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame):
        dtrain = lgb.Dataset(train[FEATURES], label=train["target"].astype(int))
        dvalid = lgb.Dataset(valid[FEATURES], label=valid["target"].astype(int))
        self.model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=self.max_rounds,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(self.early_stopping_rounds),
                lgb.log_evaluation(50),
            ],
        )
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        p = self.model.predict(
            df[FEATURES], num_iteration=self.model.best_iteration)
        p = np.clip(p, 1e-6, 1-1e-6)
        return np.vstack([1 - p, p]).T
