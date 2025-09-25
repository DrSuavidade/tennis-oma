import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

st.set_page_config(page_title="T-OMA: Tennis Probabilities", layout="wide")

st.title("🎾 T-OMA — Tennis Outcome Modeling (ATP/WTA)")

data_dir = Path("data/processed")
res_file = data_dir / "backtest_results.csv"
oof_file = data_dir / "oof_valid.csv"

col1, col2 = st.columns([1,2])

with col1:
    st.header("Backtest Folds")
    if res_file.exists():
        res = pd.read_csv(res_file)
        st.dataframe(res)
    else:
        st.info("Run the CLI to generate backtest results.")

with col2:
    st.header("Calibration & Reliability")
    if oof_file.exists():
        oof = pd.read_csv(oof_file, parse_dates=["date"])
        pcols = [c for c in oof.columns if c.startswith("p_")]
        pick = st.selectbox("Choose model probability for curve:", pcols, index=0)
        y = oof["target"].to_numpy()
        p = oof[pick].to_numpy()

        # Reliability curve (no seaborn, single plot, no custom colors)
        bins = np.linspace(0,1,11)
        inds = np.digitize(p, bins) - 1
        xs, ys = [], []
        for b in range(10):
            m = inds == b
            if m.any():
                xs.append(p[m].mean())
                ys.append(y[m].mean())

        fig = plt.figure()
        plt.plot([0,1],[0,1])
        plt.scatter(xs, ys)
        plt.xlabel("Predicted Probability")
        plt.ylabel("Empirical Win Rate")
        plt.title("Reliability Curve")
        st.pyplot(fig)

        st.subheader("Sharpness (histogram)")
        fig2 = plt.figure()
        plt.hist(p, bins=20)
        plt.xlabel("Predicted Probability")
        plt.ylabel("Count")
        plt.title("Sharpness")
        st.pyplot(fig2)
    else:
        st.info("Run the CLI to produce OOF predictions first.")

st.header("Per-Segment Metrics")
if oof_file.exists():
    oof = pd.read_csv(oof_file, parse_dates=["date"])
    oof["year"] = oof["date"].dt.year
    pcols = [c for c in oof.columns if c.startswith("p_")]
    pick = st.selectbox("Model for segment metrics", pcols, index=0, key="segpick")
    def log_loss(y_true, p):
        eps=1e-15
        p = np.clip(p, eps, 1-eps)
        return -(y_true*np.log(p) + (1-y_true)*np.log(1-p)).mean()
    seg = (
        oof.groupby(["year"])
            .apply(
                lambda g: pd.Series({
                    "log_loss": log_loss(g["target"].to_numpy(), g[pick].to_numpy()),
                    "brier": ((g[pick] - g["target"])**2).mean(),
                }),
                include_groups=False,   # <-- silences the FutureWarning
            )
            .reset_index()
    )
    st.dataframe(seg)
else:
    st.info("No OOF predictions yet.")
