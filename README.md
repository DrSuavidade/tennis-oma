# T-OMA — Tennis Outcome Modeling Architect

Predict pre-match tennis win probabilities with strong calibration and strict, time-aware backtesting.
Targets Python **3.12.11**.

## Quickstart

```bash
# 1) Create & activate env (example with uv or venv)
uv venv -p 3.12.11 .venv
.venv\Scripts\activate
uv pip install -e .

# If you prefer pip:
# python3.12 -m venv .venv && source .venv/bin/activate
# pip install -e .

# 2) Run basic pipeline on synthetic data
python -m tennis.cli build-features --asof 2018-01-01
python -m tennis.cli train --folds 3 --models elo,gbdt --tour both
python -m tennis.cli stack --since 2018-01-01 --use-market no
python -m tennis.cli evaluate --since 2018-01-01

# 3) Launch UI
streamlit run app/streamlit_app.py
```

## Data sources
Point the ingestors to your local clones/exports of e.g.
- Jeff Sackmann: https://github.com/JeffSackmann/tennis_atp
- Tennismylife DB: https://github.com/Tennismylife/TML-Database
- tennis-crystal-ball: https://github.com/mcekovic/tennis-crystal-ball

By default, this repo ships with a small synthetic dataset for smoke tests under `data/raw/synth/`.

## Structure
```
tennis-oma/
├─ pyproject.toml
├─ README.md
├─ .env.example
├─ configs/
│  ├─ base.yaml
│  ├─ cv.yaml
│  └─ models.yaml
├─ data/
│  ├─ raw/ (place external CSVs here; synth included)
│  └─ processed/
├─ src/tennis/
│  ├─ cli.py
│  ├─ config.py
│  ├─ utils/
│  │  ├─ logging.py
│  │  ├─ time_split.py
│  │  └─ leakage_sentry.py
│  ├─ data/
│  │  ├─ schema.py
│  │  ├─ ingest_matches.py
│  │  └─ ratings.py
│  ├─ features/
│  │  └─ builders.py
│  ├─ models/
│  │  ├─ registry.py
│  │  ├─ elo.py
│  │  ├─ gbdt_lgbm.py
│  │  └─ calibrate.py
│  ├─ ensemble/
│  │  └─ stacker.py
│  └─ eval/
│     ├─ metrics.py
│     └─ backtest.py
├─ app/
│  └─ streamlit_app.py
├─ scripts/
│  ├─ make_synth_data.py
│  └─ run_small_backtest.sh
└─ tests/
   ├─ test_determinism.py
   ├─ test_time_split.py
   └─ test_walkforward_smoke.py
```

## Notes
- Strict time-aware splits; no data leakage.
- Odds can be included as features (opt-in) and are always benchmarked separately.
- Retirements/WOs excluded from training by default; sensitivity toggle in configs.
- Reproducibility: fixed seeds, pinned deps in `pyproject.toml`.
