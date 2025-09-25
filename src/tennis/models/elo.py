import pandas as pd
import numpy as np
from ..data.ratings import update_elo

class EloModel:
    def __init__(self, k_base=32.0, surface_offset=40.0, decay_half_life_days=365):
        self.k = k_base
        self.surface_offset = surface_offset
        self.half_life = decay_half_life_days
        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
        # Elo updates are computed directly in feature pipeline; no separate fitting needed.
        self.is_fitted = True
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        # Use expected from elo_exp_a if present; else logistic on rating diff
        if "elo_exp_a" in df.columns:
            p = df["elo_exp_a"].clip(1e-6, 1-1e-6).to_numpy()
        else:
            rd = df["elo_a_surface"] - df["elo_b_surface"]
            p = 1 / (1 + 10 ** (-(rd)/400))
        return np.vstack([1 - p, p]).T
