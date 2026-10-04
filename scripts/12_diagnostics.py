"""Diagnose the final model's test-set over-correction from existing results only (no new model is selected).

For each main target:
  - signed bias of the time-blind OLS on GPQA in dev CV and on test (how much correction was needed)
  - the final model's time contribution on each test model: prediction with the fitted time term minus the
    prediction with the time value held at the training mean, both from the same fitted model
  - how far test models lie beyond the training time range
Output: results/diagnostics/<target>.csv and summary.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C
from src.models import Linear

OUT = C.RESULTS / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)
summ = []
for tgt in C.TARGETS:
    t = pd.read_csv(C.PROCESSED / f"table_{tgt}.csv", index_col=0, parse_dates=["release_date"])
    test = [l for l in (C.SPLITS / f"{tgt}_test_models.txt").read_text().splitlines() if l and not l.startswith("#")]
    alpha = pd.read_csv(C.RESULTS / "cv" / tgt / "summary.csv").set_index("config").loc["ridge_C_free_controls", "alpha"]
    cv = pd.read_csv(C.RESULTS / "cv" / tgt / "predictions.csv")
    te_pred = pd.read_csv(C.RESULTS / "test" / tgt / "predictions.csv")
    rows = []
    for m in t.loc[test].sort_values("release_date").index:
        d0 = t.loc[m, "release_date"]
        tr = t[t.release_date < d0]
        model = Linear("C", alpha, free_controls=True).fit(tr)
        one = t.loc[[m]]
        held = one.copy()
        held["t_years"] = tr.t_years.mean()
        p_full, p_held = model.predict(one)[0], model.predict(held)[0]
        rows.append(dict(model=m, release_date=d0, y=one.y.iloc[0], pred=p_full, time_contribution_pp=p_full - p_held,
                         years_beyond_train_max=one.t_years.iloc[0] - tr.t_years.max(),
                         years_from_train_mean=one.t_years.iloc[0] - tr.t_years.mean()))
    D = pd.DataFrame(rows)
    D.to_csv(OUT / f"{tgt}.csv", index=False)
    ols_cv = cv[cv.config == "single_ols_gpqa"].err.mean()
    ols_te = te_pred[te_pred.config == "single_ols_gpqa"].err.mean()
    fin_te = te_pred[te_pred.config == "ridge_C_free_controls"].err.mean()
    notime_te = te_pred[te_pred.config == "ridge_C_minus_t_years"].err.mean()
    summ.append(dict(target=tgt, ols_gpqa_bias_dev_cv=ols_cv, ols_gpqa_bias_test=ols_te,
                     final_bias_test=fin_te, final_minus_time_bias_test=notime_te,
                     mean_time_contribution_test_pp=D.time_contribution_pp.mean(),
                     mean_years_from_train_mean=D.years_from_train_mean.mean(),
                     max_years_beyond_train_max=D.years_beyond_train_max.max()))
S = pd.DataFrame(summ)
S.to_csv(OUT / "summary.csv", index=False)
print(S.round(2).T.to_string())
