"""Final model: ridge_C_free_controls (Set C, time and source unpenalized); ablations are relative to it.
Time-series cross-validation on the DEV set of a target (test models never touched).

Folds: dev models sorted by release date; TimeSeriesSplit-style expanding window with N_FOLDS
validation blocks of VAL_SIZE models each. Training rows are strictly older than the block's first
release date (same-day releases are never used to predict each other).

Usage: python scripts/05_cv.py [target]   (default: config.MAIN_TARGET)
Outputs results/cv/<target>/: predictions.csv, tuning.csv, summary.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import benchpress_baseline as bp
from src import config as C
from src.data import load_models
from src.models import GBM, Linear, SingleOLS, TrainMean

# Decision threshold (pp): one task-sampling standard error at the dev-set mean (see paper §6).
THRESHOLD_PP = {"terminal_bench": 5.3, "swe_bench_pro": 1.8}

TGT = sys.argv[1] if len(sys.argv) > 1 else C.MAIN_TARGET
N_FOLDS, VAL_SIZE = 4, 6  # 24 validation predictions; smallest training fold = 16 models
ALPHAS = np.logspace(-3, 3, 13)
KS = [1, 2, 3, 5, 10]
OUT = C.RESULTS / "cv" / TGT
OUT.mkdir(parents=True, exist_ok=True)

t = pd.read_csv(C.PROCESSED / f"table_{TGT}.csv", index_col=0, parse_dates=["release_date"])
test = {l for l in (C.SPLITS / f"{TGT}_test_models.txt").read_text().splitlines() if l and not l.startswith("#")}
dev = t[~t.index.isin(test)].sort_values(["release_date", "model_name"])
W = pd.read_csv(C.PROCESSED / "score_matrix_verified.csv", index_col=0)
rel = load_models().release_date
W = W.loc[W.index.intersection(rel.index)].drop(index=list(test), errors="ignore")

folds = []
for f in range(N_FOLDS):
    start = len(dev) - (N_FOLDS - f) * VAL_SIZE
    val = dev.iloc[start:start + VAL_SIZE]
    tr = dev[dev.release_date < val.release_date.min()]
    folds.append((f, tr, val))
    print(f"fold {f}: train {len(tr)} (to {tr.release_date.max():%Y-%m-%d}), "
          f"val {len(val)} ({val.release_date.min():%Y-%m-%d} to {val.release_date.max():%Y-%m-%d})")

rows = []


def record(config, fold, val, pred, **hp):
    for m, yhat in zip(val.index, pred):
        rows.append(dict(config=config, fold=fold, model=m, y=val.loc[m, "y"], pred=yhat, **hp))


for f, tr, val in folds:
    record("mean", f, val, TrainMean().fit(tr).predict(val))
    record("single_ols_gpqa", f, val, SingleOLS().fit(tr).predict(val))
    record("gbm", f, val, GBM().fit(tr).predict(val))
    past = W[rel.loc[W.index] < val.release_date.min()]
    probes = W.loc[val.index, [p for p in C.PROBES if p in W.columns]]
    for meth in bp.METHODS:
        record(meth, f, val, bp.predict_individually(past, probes, TGT, method=meth).to_numpy())
    for a in ALPHAS:
        record("ridge_A", f, val, Linear("A", a).fit(tr).predict(val), alpha=a)
        record("ridge_C", f, val, Linear("C", a).fit(tr).predict(val), alpha=a)
        record("ridge_C_free_controls", f, val, Linear("C", a, free_controls=True).fit(tr).predict(val), alpha=a)
        # group ablations: drop one engineered group from Set C
        for g in ["t_years", "third_party", "missing"]:
            record(f"ridge_C_minus_{g}", f, val, Linear("C", a, drop=g, free_controls=True).fit(tr).predict(val), alpha=a)
        for k in KS:
            record("ridge_B", f, val, Linear("B", a, k=k).fit(tr).predict(val), alpha=a, k=k)
    print(f"fold {f} done")

P = pd.DataFrame(rows)
P["err"] = P.pred - P.y
P.to_csv(OUT / "predictions.csv", index=False)

# hyperparameter selection by mean-over-folds MAE (validation folds of the dev set only)
hp_cols = ["alpha", "k"]
for c in hp_cols:
    if c not in P:
        P[c] = np.nan
fold_mae = (P.assign(ae=P.err.abs()).groupby(["config", *hp_cols, "fold"], dropna=False).ae.mean()
            .groupby(["config", *hp_cols], dropna=False).agg(["mean", "std"]).reset_index())
fold_mae.to_csv(OUT / "tuning.csv", index=False)
ok = fold_mae.dropna(subset=["mean"])
best = ok.loc[ok.groupby("config")["mean"].idxmin()]

summ = []
for _, b in best.iterrows():
    sel = P[P.config == b.config]
    for c in hp_cols:
        sel = sel[sel[c].isna()] if pd.isna(b[c]) else sel[np.isclose(sel[c], b[c])]
    per_fold = sel.groupby("fold").err.apply(lambda e: e.abs().mean())
    summ.append(dict(config=b.config, alpha=b.alpha, k=b.k, n=len(sel),
                     coverage=sel.pred.notna().mean(), cv_mae_mean=per_fold.mean(), cv_mae_std=per_fold.std(),
                     rmse=np.sqrt((sel.err ** 2).mean()), bias=sel.err.mean(),
                     share_within_threshold=(sel.err.abs() <= THRESHOLD_PP[TGT]).mean()))
S = pd.DataFrame(summ).sort_values("cv_mae_mean")
S.to_csv(OUT / "summary.csv", index=False)
print(S.round(3).to_string(index=False))
