"""Development cross-validation pooled over the benchmarks (development splits only; test models are dropped).

Terminal-Bench 2.0 and SWE-Bench Pro: the selected-hyperparameter predictions of scripts/05_cv.py.
The other 18: the same four-block expanding-window split and hyperparameter grids as
scripts/09_generalization_full.py; each method contributes the CV predictions at its selected hyperparameters.
Every benchmark has at least 12 development models (scripts/07_generalization_freeze.py).

Output: results/pooled_cv/{predictions,per_target,pooled}.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_1samp, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import benchpress_baseline as bp
from src import config as C
from src.data import load_models
from src.models import GBM, Linear, SingleOLS, TrainMean

ALPHAS = np.logspace(-2, 3, 11)
KS = [1, 2, 3, 5, 10]
OUT = C.RESULTS / "pooled_cv"
OUT.mkdir(parents=True, exist_ok=True)

MAIN_NAMES = {"mean": "Training mean", "single_ols_gpqa": "OLS on GPQA", "gbm": "GBM", "benchpress": "BenchPress",
              "soft_impute": "Soft-Impute", "ridge_C_free_controls": "TimeRidge", "ridge_C_minus_t_years": "TimeRidge - time",
              "ridge_A": "Set A (raw)", "ridge_B": "Set B (top-k)", "ridge_C": "Set C, all shrunk",
              "ridge_C_minus_third_party": "TimeRidge - source", "ridge_C_minus_missing": "TimeRidge - missing ind."}
FIXED = {"Training mean": lambda: TrainMean(), "OLS on GPQA": lambda: SingleOLS(), "GBM": lambda: GBM()}
RIDGE = {
    "TimeRidge": (lambda h: Linear("C", h["alpha"], free_controls=True), [dict(alpha=a) for a in ALPHAS]),
    "TimeRidge - time": (lambda h: Linear("C", h["alpha"], drop="t_years", free_controls=True), [dict(alpha=a) for a in ALPHAS]),
    "Set A (raw)": (lambda h: Linear("A", h["alpha"]), [dict(alpha=a) for a in ALPHAS]),
    "Set B (top-k)": (lambda h: Linear("B", h["alpha"], k=h["k"]), [dict(alpha=a, k=k) for a in ALPHAS for k in KS]),
    "Set C, all shrunk": (lambda h: Linear("C", h["alpha"]), [dict(alpha=a) for a in ALPHAS]),
    "TimeRidge - source": (lambda h: Linear("C", h["alpha"], drop="third_party", free_controls=True), [dict(alpha=a) for a in ALPHAS]),
    "TimeRidge - missing ind.": (lambda h: Linear("C", h["alpha"], drop="missing", free_controls=True), [dict(alpha=a) for a in ALPHAS]),
}

parts = []
# the two benchmarks with a full CV in scripts/05_cv.py
for tgt in C.TARGETS:
    P = pd.read_csv(C.RESULTS / "cv" / tgt / "predictions.csv")
    S = pd.read_csv(C.RESULTS / "cv" / tgt / "summary.csv").set_index("config")
    for cfg, name in MAIN_NAMES.items():
        x = P[P.config == cfg]
        for h in ("alpha", "k"):
            v = S.loc[cfg, h]
            x = x[x[h].isna()] if pd.isna(v) else x[np.isclose(x[h], v)]
        parts.append(pd.DataFrame(dict(target=tgt, method=name, model=x.model, y=x.y, pred=x.pred)))

W = pd.read_csv(C.PROCESSED / "score_matrix_verified.csv", index_col=0)
rel = load_models().release_date
W = W.loc[W.index.intersection(rel.index)]
probes = [p for p in C.PROBES if p in W.columns]
for f in sorted((C.SPLITS / "generalization").glob("*_test_models.txt")):
    tgt = f.name.removesuffix("_test_models.txt")
    t = pd.read_csv(C.PROCESSED / f"gen_table_{tgt}.csv", index_col=0, parse_dates=["release_date"])
    test = [l for l in f.read_text().splitlines() if l and not l.startswith("#")]
    dev = t[~t.index.isin(test)].sort_values(["release_date", "model_name"])
    Wd = W.drop(index=test, errors="ignore")
    size = max(3, round(0.15 * len(dev)))
    folds = []
    for k in range(4):
        start = len(dev) - (4 - k) * size
        val = dev.iloc[start:start + size]
        tr = dev[dev.release_date < val.release_date.min()]
        if len(tr) >= 4:
            folds.append((tr, val))
    rows = []

    def add(name, val, pred):
        rows.append(pd.DataFrame(dict(target=tgt, method=name, model=val.index, y=val.y.to_numpy(), pred=pred)))

    for tr, val in folds:
        for name, make in FIXED.items():
            add(name, val, make().fit(tr).predict(val))
        past = Wd[rel.loc[Wd.index] < val.release_date.min()]
        for meth, name in (("benchpress", "BenchPress"), ("soft_impute", "Soft-Impute")):
            add(name, val, bp.predict_individually(past, Wd.loc[val.index, probes], tgt, method=meth).to_numpy())
    for name, (make, grid) in RIDGE.items():
        best, best_mae = None, np.inf
        for h in grid:
            preds = [pd.DataFrame(dict(model=val.index, y=val.y.to_numpy(), pred=make(h).fit(tr).predict(val)))
                     for tr, val in folds]
            d = pd.concat(preds)
            mae = (d.pred - d.y).abs().mean()
            if mae < best_mae:
                best, best_mae = d, mae
        rows.append(best.assign(target=tgt, method=name))
    parts.extend(rows)
    print(f"{tgt}: dev {len(dev)}, {len(folds)} folds, block {size}")

P = pd.concat(parts, ignore_index=True)
P["err"] = P.pred - P.y
P.to_csv(OUT / "predictions.csv", index=False)
per = P.groupby(["target", "method"]).agg(n=("err", "size"), bias=("err", "mean"),
                                         mae=("err", lambda e: e.abs().mean())).reset_index()
per.to_csv(OUT / "per_target.csv", index=False)
ref = P[P.method == "TimeRidge"].set_index(["target", "model"]).err.abs()
out = []
for meth, g in P.groupby("method"):
    b = per[per.method == meth]
    a = g.set_index(["target", "model"]).err.abs().reindex(ref.index)
    under = int((b.bias < 0).sum())
    out.append(dict(method=meth, benchmarks=len(b), under=under, sign_test_p=binomtest(under, len(b), 0.5, alternative="greater").pvalue,
                    n_predictions=len(g), pooled_bias=g.err.mean(), bias_p_two_sided=ttest_1samp(g.err, 0).pvalue,
                    pooled_mae=g.err.abs().mean(), pooled_rmse=np.sqrt((g.err ** 2).mean()),
                    wilcoxon_p_vs_timeridge=np.nan if meth == "TimeRidge" else wilcoxon(ref, a).pvalue))
pooled = pd.DataFrame(out).sort_values("pooled_mae")
pooled.to_csv(OUT / "pooled.csv", index=False)
print(pooled.round(3).to_string(index=False))
