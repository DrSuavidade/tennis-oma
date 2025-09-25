import pandas as pd
import numpy as np
from ..utils.time_split import rolling_time_splits
from ..models.registry import make_model
from ..models.calibrate import Calibrator
from ..eval.metrics import log_loss, brier, ece

def walk_forward(df, cfg_models, cv_cfg, seed=42):
    df = df.copy().sort_values("date")
    folds = rolling_time_splits(df, "date", cv_cfg["freq"], cv_cfg["n_folds"],
                                cv_cfg["train_min_weeks"], cv_cfg["valid_weeks"], cv_cfg["test_weeks"])

    results = []
    oof_rows = []
    for i, (tr, va, te, vs, ve) in enumerate(folds):
        train = df.loc[tr]
        valid = df.loc[va]
        test  = df.loc[te]
        preds_valid = {"target": valid["target"].values}
        preds_test  = {"target": test["target"].values}

        for name, params in cfg_models.items():
            model = make_model(name, params)
            if name == "elo":
                # No fit needed
                model.fit(train)
            else:
                model.fit(train, valid)
            pv = model.predict_proba(valid)[:,1]
            pt = model.predict_proba(test)[:,1]
            preds_valid[f"p_{name}"] = pv
            preds_test[f"p_{name}"] = pt

        # choose calibrator
        cal = Calibrator(method="auto")
        # Stack simple average for calibration target
        pv_avg = np.mean([preds_valid[c] for c in preds_valid if c.startswith("p_")], axis=0)
        cal.fit(pv_avg, preds_valid["target"])
        pt_avg = np.mean([preds_test[c] for c in preds_test if c.startswith("p_")], axis=0)
        pt_cal = cal.transform(pt_avg)

        ll = log_loss(preds_test["target"], pt_cal)
        br = brier(preds_test["target"], pt_cal)
        ec = ece(preds_test["target"], pt_cal, n_bins=10)
        results.append({"fold": i, "valid_start": str(vs.date()), "log_loss": ll, "brier": br, "ece": ec})

        oof_valid = {"date": valid["date"].values, **preds_valid}
        oof_rows.append(pd.DataFrame(oof_valid))

    return pd.DataFrame(results), pd.concat(oof_rows, ignore_index=True)
