import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

class OOFStacker:
    def __init__(self, use_market: bool = False):
        self.use_market = use_market
        self.lr = LogisticRegression(max_iter=1000)

    def fit(self, oof_df: pd.DataFrame):
        cols = [c for c in oof_df.columns if c.startswith("p_")]
        if self.use_market and "market_prob_a" in oof_df.columns:
            cols += ["market_prob_a"]
        X = oof_df[cols].to_numpy()
        y = oof_df["target"].to_numpy()
        self.lr.fit(X, y)
        return self

    def predict(self, preds_df: pd.DataFrame) -> np.ndarray:
        cols = [c for c in preds_df.columns if c.startswith("p_")]
        if self.use_market and "market_prob_a" in preds_df.columns:
            cols += ["market_prob_a"]
        X = preds_df[cols].to_numpy()
        p = self.lr.predict_proba(X)[:,1]
        return np.clip(p, 1e-6, 1-1e-6)
