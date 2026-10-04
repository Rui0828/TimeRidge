"""Generalization check on every eligible expensive benchmark (notes/generalization_rule.md). Run once.

Same final design as the main targets. Output: results/generalization/{predictions,per_target,pooled}.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_1samp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import benchpress_baseline as bp
from src import config as C
from src.data import load_models
from src.models import Linear, SingleOLS

ALPHAS = np.logspace(-2, 3, 11)
OUT = C.RESULTS / "generalization"
if OUT.exists() and "--rerun-identical" not in sys.argv:
    sys.exit("results/generalization/ exists. Pass --rerun-identical only to reproduce the same numbers.")
OUT.mkdir(parents=True, exist_ok=True)

W = pd.read_csv(C.PROCESSED / "score_matrix_verified.csv", index_col=0)
rel = load_models().release_date
W = W.loc[W.index.intersection(rel.index)]
targets = sorted(f.name.removesuffix("_test_models.txt") for f in (C.SPLITS / "generalization").glob("*_test_models.txt"))


def final(a, time=True):
    return Linear("C", a, free_controls=True, drop=None if time else "t_years")


rows = []
for tgt in targets:
    t = pd.read_csv(C.PROCESSED / f"gen_table_{tgt}.csv", index_col=0, parse_dates=["release_date"])
    test = [l for l in (C.SPLITS / "generalization" / f"{tgt}_test_models.txt").read_text().splitlines()
            if l and not l.startswith("#")]
    dev = t[~t.index.isin(test)].sort_values(["release_date", "model_name"])
    alpha = {}
    for time in (True, False):
        if len(dev) < 12:
            alpha[time] = 100.0
            continue
        size = max(3, round(0.15 * len(dev)))
        cv = {}
        for a in ALPHAS:
            errs = []
            for f in range(4):
                start = len(dev) - (4 - f) * size
                val = dev.iloc[start:start + size]
                tr = dev[dev.release_date < val.release_date.min()]
                if len(tr) >= 4:
                    errs.append(final(a, time).fit(tr).predict(val) - val.y.to_numpy())
            cv[a] = np.abs(np.concatenate(errs)).mean()
        alpha[time] = min(cv, key=cv.get)
    te = t.loc[test].sort_values(["release_date", "model_name"])
    for m in te.index:
        d0 = te.loc[m, "release_date"]
        tr = t[t.release_date < d0]
        one = te.loc[[m]]
        preds = {"Final model": final(alpha[True]).fit(tr).predict(one)[0],
                 "Final model - time": final(alpha[False], False).fit(tr).predict(one)[0],
                 "OLS on GPQA": SingleOLS().fit(tr).predict(one)[0]}
        past = W[rel.loc[W.index] < d0]
        preds["BenchPress"] = bp.predict(past, W.loc[[m], [p for p in C.PROBES if p in W.columns]], tgt).iloc[0]
        for k, p in preds.items():
            rows.append(dict(target=tgt, method=k, model=m, release_date=d0, y=one.y.iloc[0], pred=p, n_train=len(tr)))
    print(f"{tgt}: dev {len(dev)}, test {len(te)}, alpha {alpha[True]:.3g} / {alpha[False]:.3g}")

P = pd.DataFrame(rows)
P["err"] = P.pred - P.y
P.to_csv(OUT / "predictions.csv", index=False)
per = P.groupby(["target", "method"]).agg(n=("err", "size"), bias=("err", "mean"),
                                         mae=("err", lambda e: e.abs().mean())).reset_index()
per.to_csv(OUT / "per_target.csv", index=False)
pooled = []
for meth, g in P.groupby("method"):
    b = per[per.method == meth]
    neg = int((b.bias < 0).sum())
    pooled.append(dict(method=meth, targets=len(b), targets_negative_bias=neg,
                       sign_test_p=binomtest(neg, len(b), 0.5, alternative="greater").pvalue,
                       n_points=len(g), pooled_bias=g.err.mean(),
                       pooled_bias_p_lt0=ttest_1samp(g.err, 0, alternative="less").pvalue,
                       pooled_mae=g.err.abs().mean()))
pooled = pd.DataFrame(pooled)
pooled.to_csv(OUT / "pooled.csv", index=False)
print(per.pivot(index="target", columns="method", values="bias").round(1).to_string())
print(pooled.round(4).to_string(index=False))
