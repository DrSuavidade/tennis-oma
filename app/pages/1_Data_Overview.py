import streamlit as st
import pandas as pd
from pathlib import Path

st.title("Data Overview & QA")

asof = st.text_input("as-of (folder under data/processed/)", value="")
if not asof:
    st.info("Enter as-of like 2024-12-31")
    st.stop()

base = Path(f"data/processed/asof={asof}")
if not base.exists():
    st.error(f"Missing folder: {base}")
    st.stop()

matches = pd.read_parquet(base / "cur_matches.parquet")
st.write("Curated matches:", matches.shape)
st.dataframe(matches.head(50))

if (base.parent.parent / "qa").exists():
    qa_files = sorted((base.parent.parent / "qa").glob("issues_*.json"))
    if qa_files:
        st.subheader("Validation Issues")
        for f in qa_files:
            st.json(pd.read_json(f).to_dict(orient="records"))
