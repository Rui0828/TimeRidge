"""Complete the generalization check: every Table 3 baseline and Table 4 ablation on the 18 frozen benchmarks.

Rule: notes/generalization_full_rule.md (written before this script ran). The 19 benchmarks, their frozen test
splits and the TimeRidge design are unchanged; the four methods already evaluated by scripts/08_generalization.py
are read from results/generalization/predictions.csv, not rerun. Run once.

Output: results/generalization_full/{predictions,per_target,pooled}.csv
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
from src.models import GBM, Linear, TrainMean

ALPHAS = np.logspace(-2, 3, 11)  # identical to scripts/08_generalization.py
KS = [1, 2, 3, 5, 10]
SMALL_DEV = dict(alpha=100.0, k=5)
OUT = C.RESULTS / "generalization_full"
if OUT.exists() and "--rerun-identical" not in sys.argv:
    sys.exit("results/generalization_full/ exists. Pass --rerun-identical only to reproduce the same numbers.")
OUT.mkdir(parents=True, exist_ok=True)

# tuned ridge variants: name -> (constructor from hyperparameters, grid of hyperparameter dicts)
RIDGE = {
    "Set A (raw)": (lambda h: Linear("A", h["alpha"]), [dict(alpha=a) for a in ALPHAS]),
    "Set B (top-k)": (lambda h: Linear("B", h["alpha"], k=h["k"]), [dict(alpha=a, k=k) for a in ALPHAS for k in KS]),
    "Set C, all shrunk": (lambda h: Linear("C", h["alpha"]), [dict(alpha=a) for a in ALPHAS]),
    "TimeRidge - source": (lambda h: Linear("C", h["alpha"], drop="third_party", free_controls=True),
                           [dict(alpha=a) for a in ALPHAS]),
    "TimeRidge - missing ind.": (lambda h: Linear("C", h["alpha"], drop="missing", free_controls=True),
                                 [dict(alpha=a) for a in ALPHAS]),
}

W = pd.read_csv(C.PROCESSED / "score_matrix_verified.csv", index_col=0)
rel = load_models().release_date
W = W.loc[W.index.intersection(rel.index)]
probes = [p for p in C.PROBES if p in W.columns]
targets = sorted(f.name.removesuffix("_test_models.txt") for f in (C.SPLITS / "generalization").glob("*_test_models.txt"))


def tune(make, grid, dev):
    if len(dev) < 12:
        return {k: SMALL_DEV[k] for k in grid[0]}
    size = max(3, round(0.15 * len(dev)))
    best, best_mae = None, np.inf
    for h in grid:
        errs = []
        for f in range(4):
            start = len(dev) - (4 - f) * size
            val = dev.iloc[start:start + size]
            tr = dev[dev.release_date < val.release_date.min()]
            if len(tr) >= 4:
                errs.append(make(h).fit(tr).predict(val) - val.y.to_numpy())
        mae = np.abs(np.concatenate(errs)).mean()
        if mae < best_mae:
            best, best_mae = h, mae
    return best


rows = []
for tgt in targets:
    t = pd.read_csv(C.PROCESSED / f"gen_table_{tgt}.csv", index_col=0, parse_dates=["release_date"])
    test = [l for l in (C.SPLITS / "generalization" / f"{tgt}_test_models.txt").read_text().splitlines()
            if l and not l.startswith("#")]
    dev = t[~t.index.isin(test)].sort_values(["release_date", "model_name"])
    hp = {name: tune(make, grid, dev) for name, (make, grid) in RIDGE.items()}
    te = t.loc[test].sort_values(["release_date", "model_name"])
    for m in te.index:
        d0 = te.loc[m, "release_date"]
        tr = t[t.release_date < d0]
        one = te.loc[[m]]
        preds = {"Training mean": TrainMean().fit(tr).predict(one)[0],
                 "GBM": GBM().fit(tr).predict(one)[0]}
        past = W[rel.loc[W.index] < d0]
        preds["Soft-Impute"] = bp.predict(past, W.loc[[m], probes], tgt, method="soft_impute").iloc[0]
        for name, (make, _) in RIDGE.items():
            preds[name] = make(hp[name]).fit(tr).predict(one)[0]
        for k, p in preds.items():
            rows.append(dict(target=tgt, method=k, model=m, release_date=d0, y=one.y.iloc[0], pred=p, n_train=len(tr)))
    print(f"{tgt}: dev {len(dev)}, test {len(te)}, hp {hp}")

new = pd.DataFrame(rows)
new["err"] = new.pred - new.y
old = pd.read_csv(C.RESULTS / "generalization" / "predictions.csv", parse_dates=["release_date"])
old["method"] = old.method.replace({"Final model": "TimeRidge", "Final model - time": "TimeRidge - time"})
P = pd.concat([old, new], ignore_index=True)
P.to_csv(OUT / "predictions.csv", index=False)

per = P.groupby(["target", "method"]).agg(n=("err", "size"), bias=("err", "mean"),
                                         mae=("err", lambda e: e.abs().mean()),
                                         rmse=("err", lambda e: np.sqrt((e ** 2).mean()))).reset_index()
per.to_csv(OUT / "per_target.csv", index=False)
ref = P[P.method == "TimeRidge"].set_index(["target", "model"]).err.abs()
pooled = []
for meth, g in P.groupby("method"):
    b = per[per.method == meth]
    neg, pos = int((b.bias < 0).sum()), int((b.bias > 0).sum())
    a = g.set_index(["target", "model"]).err.abs().reindex(ref.index)
    pooled.append(dict(method=meth, targets=len(b), under=neg, over=pos,
                       sign_test_p=binomtest(neg, len(b), 0.5, alternative="greater").pvalue,
                       n_points=len(g), pooled_bias=g.err.mean(),
                       pooled_bias_p_lt0=ttest_1samp(g.err, 0, alternative="less").pvalue,
                       pooled_mae=g.err.abs().mean(), pooled_rmse=np.sqrt((g.err ** 2).mean()),
                       wilcoxon_p_vs_timeridge=np.nan if meth == "TimeRidge" else wilcoxon(ref, a).pvalue))
pooled = pd.DataFrame(pooled).sort_values("pooled_mae")
pooled.to_csv(OUT / "pooled.csv", index=False)
print(pooled.round(3).to_string(index=False))
