import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

class Calibrator:
    def __init__(self, method: str = "auto"):
        self.method = method
        self.model = None

    def fit(self, p: np.ndarray, y: np.ndarray):
        if self.method == "platt" or (self.method == "auto" and len(np.unique(y)) == 2):
            lr = LogisticRegression()
            lr.fit(p.reshape(-1,1), y)
            self.model = ("platt", lr)
        else:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p, y)
            self.model = ("isotonic", iso)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        kind, m = self.model
        if kind == "platt":
            return m.predict_proba(p.reshape(-1,1))[:,1]
        else:
            return m.transform(p)
