"""Pool the test predictions of all 20 agentic benchmarks: Terminal-Bench 2.0 and SWE-Bench Pro (results/test/) and
the other 18 (results/generalization_full/). No model is fit here.

Output: results/pooled_test/{predictions,per_target,pooled}.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_1samp, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C

OUT = C.RESULTS / "pooled_test"
OUT.mkdir(parents=True, exist_ok=True)

MAIN_NAMES = {"mean": "Training mean", "single_ols_gpqa": "OLS on GPQA", "gbm": "GBM", "benchpress": "BenchPress",
              "soft_impute": "Soft-Impute", "ridge_C_free_controls": "TimeRidge", "ridge_C_minus_t_years": "TimeRidge - time",
              "ridge_A": "Set A (raw)", "ridge_B": "Set B (top-k)", "ridge_C": "Set C, all shrunk",
              "ridge_C_minus_third_party": "TimeRidge - source", "ridge_C_minus_missing": "TimeRidge - missing ind."}
METHODS = list(MAIN_NAMES.values())

parts = []
for tgt in C.TARGETS:
    P = pd.read_csv(C.RESULTS / "test" / tgt / "predictions.csv")
    P = P[P.config.isin(MAIN_NAMES)].assign(target=tgt, method=lambda d: d.config.map(MAIN_NAMES))
    parts.append(P[["target", "method", "model", "y", "pred"]])
G = pd.read_csv(C.RESULTS / "generalization_full" / "predictions.csv")
parts.append(G[G.method.isin(METHODS)][["target", "method", "model", "y", "pred"]])
P = pd.concat(parts, ignore_index=True)
P["err"] = P.pred - P.y
assert P.groupby("method").target.nunique().eq(20).all(), P.groupby("method").target.nunique()
P.to_csv(OUT / "predictions.csv", index=False)

per = P.groupby(["target", "method"]).agg(n=("err", "size"), bias=("err", "mean"),
                                         mae=("err", lambda e: e.abs().mean()),
                                         rmse=("err", lambda e: np.sqrt((e ** 2).mean()))).reset_index()
per.to_csv(OUT / "per_target.csv", index=False)

ref = P[P.method == "TimeRidge"].set_index(["target", "model"]).err.abs()
rows = []
for meth in METHODS:
    g = P[P.method == meth]
    b = per[per.method == meth]
    a = g.set_index(["target", "model"]).err.abs().reindex(ref.index)
    under = int((b.bias < 0).sum())
    rows.append(dict(method=meth, benchmarks=len(b), under=under, over=int((b.bias > 0).sum()),
                     sign_test_p=binomtest(under, len(b), 0.5, alternative="greater").pvalue,
                     n_models=len(g), pooled_bias=g.err.mean(),
                     bias_p_two_sided=ttest_1samp(g.err, 0).pvalue,
                     pooled_mae=g.err.abs().mean(), pooled_rmse=np.sqrt((g.err ** 2).mean()),
                     wilcoxon_p_vs_timeridge=np.nan if meth == "TimeRidge" else wilcoxon(ref, a).pvalue))
pooled = pd.DataFrame(rows)
pooled.to_csv(OUT / "pooled.csv", index=False)
print(pooled.round(3).to_string(index=False))
