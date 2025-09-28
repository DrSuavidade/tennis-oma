import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

class OOFStacker:
    def __init__(self, use_market: bool = False):
        self.use_market = use_market
        self.lr = LogisticRegression(max_iter=1000)

    def fit(self, oof_df: pd.DataFrame):
        df = oof_df.copy()
        # --- normalize target name ---
        if "y_true" not in df.columns and "target" in df.columns:
            df = df.rename(columns={"target": "y_true"})
        if "y_true" not in df.columns:
            raise KeyError("OOFStacker.fit expects 'y_true' (or 'target') in the OOF frame.")

        # --- detect long vs wide ---
        is_long = ("model" in df.columns) and ("p" in df.columns)
        if is_long:
            # pivot: one column per model probability
            wide = df.pivot_table(
                index=["date", "match_id", "fold"],
                columns="model",
                values="p",
                aggfunc="mean",
            )
            # name columns p_MODEL
            wide.columns = [f"p_{c}" for c in wide.columns]
            wide = wide.reset_index()

            base = (
                df.drop_duplicates(subset=["date", "match_id", "fold"])
                  [["date", "match_id", "fold", "y_true"]]
            )
            data = base.merge(wide, on=["date", "match_id", "fold"], how="left")

            # optional market column (if your OOF long frame already carries it row-wise)
            if self.use_market and "market_prob_a" in df.columns:
                mkt = (
                    df.groupby(["date", "match_id", "fold"], as_index=False)
                      ["market_prob_a"].mean()
                )
                data = data.merge(mkt, on=["date", "match_id", "fold"], how="left")
        else:
            # already wide: expect p_* columns
            data = df.copy()

        # --- assemble X/y ---
        proba_cols = [c for c in data.columns if c.startswith("p_")]
        if self.use_market and "market_prob_a" in data.columns:
            proba_cols += ["market_prob_a"]
        if not proba_cols:
            raise ValueError("No model probability columns found. Expected 'p_*' (and/or 'market_prob_a').")

        self.feature_cols_ = proba_cols
        X = data[proba_cols].to_numpy()
        y = data["y_true"].to_numpy()
        self.lr.fit(X, y)
        return self

    def predict(self, preds_df: pd.DataFrame) -> np.ndarray:
        df = preds_df.copy()

        # Accept either long (date, match_id, model, p) or wide (p_*) inputs
        is_long = ("model" in df.columns) and ("p" in df.columns)
        if is_long:
            wide = df.pivot_table(
                index=["date", "match_id"],
                columns="model",
                values="p",
                aggfunc="mean",
            )
            wide.columns = [f"p_{c}" for c in wide.columns]
            wide = wide.reset_index()

            if self.use_market and "market_prob_a" in df.columns:
                mkt = (
                    df.groupby(["date", "match_id"], as_index=False)["market_prob_a"]
                    .mean()
                )
                wide = wide.merge(mkt, on=["date", "match_id"], how="left")
            data = wide
        else:
            data = df

        # Column order must match training
        if not hasattr(self, "feature_cols_"):
            # Back-compat: infer from available columns if fit() wasn’t called in this session
            cols = [c for c in data.columns if c.startswith("p_")]
            if self.use_market and "market_prob_a" in data.columns:
                cols += ["market_prob_a"]
            if not cols:
                raise ValueError("No model probability columns found for prediction.")
            self.feature_cols_ = cols

        # Fill any missing expected columns (e.g., a model absent in this split)
        missing = [c for c in self.feature_cols_ if c not in data.columns]
        for c in missing:
            # neutral prob; you could also raise instead if you prefer to fail fast
            data[c] = 0.5

        X = data[self.feature_cols_].to_numpy()
        p = self.lr.predict_proba(X)[:, 1]
        return np.clip(p, 1e-6, 1 - 1e-6)

