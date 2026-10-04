"""Final evaluation on the frozen TEST set, after notes/decisions.md was locked.

Protocol: test models in release order; each is predicted by models released strictly earlier
(all dev models plus earlier test models), i.e. the same new-model protocol as CV.
Hyperparameters are read from the dev CV (results/cv/<target>/summary.csv); nothing is tuned here.

Outputs results/test/<target>/: predictions.csv, metrics.csv, comparisons.csv
"""
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import benchpress_baseline as bp
from src import config as C
from src.data import load_models
from src.models import GBM, Linear, SingleOLS, TrainMean

FINAL = "ridge_C_free_controls"
BASELINES = ["mean", "single_ols_gpqa", "gbm", "benchpress", "soft_impute"]
ABLATIONS = ["ridge_A", "ridge_B", "ridge_C", "ridge_C_minus_t_years", "ridge_C_minus_third_party",
             "ridge_C_minus_missing"]
THRESHOLD_PP = {"terminal_bench": 5.3, "swe_bench_pro": 1.8}
N_BOOT = 10000

W = pd.read_csv(C.PROCESSED / "score_matrix_verified.csv", index_col=0)
rel = load_models().release_date
W = W.loc[W.index.intersection(rel.index)]


def cfg_rng(name):
    return np.random.default_rng([C.SEED, zlib.crc32(name.encode())])


def make(cfg, hp):
    a = hp.get("alpha")
    if cfg == "mean":
        return TrainMean()
    if cfg == "single_ols_gpqa":
        return SingleOLS()
    if cfg == "gbm":
        return GBM()
    if cfg == "ridge_A":
        return Linear("A", a)
    if cfg == "ridge_B":
        return Linear("B", a, k=int(hp["k"]))
    if cfg == "ridge_C":
        return Linear("C", a)
    if cfg == "ridge_C_free_controls":
        return Linear("C", a, free_controls=True)
    if cfg.startswith("ridge_C_minus_"):
        return Linear("C", a, drop=cfg.removeprefix("ridge_C_minus_"), free_controls=True)
    raise ValueError(cfg)


if (C.RESULTS / "test").exists() and "--rerun-identical" not in sys.argv:
    sys.exit("results/test/ exists. Pass --rerun-identical only to reproduce the same numbers.")

for tgt in C.TARGETS:
    out = C.RESULTS / "test" / tgt
    out.mkdir(parents=True, exist_ok=True)
    t = pd.read_csv(C.PROCESSED / f"table_{tgt}.csv", index_col=0, parse_dates=["release_date"])
    test = [l for l in (C.SPLITS / f"{tgt}_test_models.txt").read_text().splitlines() if l and not l.startswith("#")]
    S = pd.read_csv(C.RESULTS / "cv" / tgt / "summary.csv").set_index("config")
    hps = {c: S.loc[c, ["alpha", "k"]].dropna().to_dict() for c in S.index}
    te = t.loc[test].sort_values(["release_date", "model_name"])
    rows = []
    for m in te.index:
        d0 = te.loc[m, "release_date"]
        tr = t[t.release_date < d0]
        one = te.loc[[m]]
        for cfg in [FINAL, *BASELINES, *ABLATIONS]:
            if cfg in bp.METHODS:
                past = W[rel.loc[W.index] < d0]
                probes = W.loc[[m], [p for p in C.PROBES if p in W.columns]]
                pred = bp.predict(past, probes, tgt, method=cfg).iloc[0]
            else:
                pred = make(cfg, hps.get(cfg, {})).fit(tr).predict(one)[0]
            rows.append(dict(config=cfg, model=m, model_name=te.loc[m, "model_name"], release_date=d0,
                             y=te.loc[m, "y"], pred=pred, n_train=len(tr),
                             target_third_party=te.loc[m, "target_third_party"],
                             target_noncanonical=te.loc[m, "target_noncanonical"]))
    P = pd.DataFrame(rows)
    P["err"] = P.pred - P.y
    P.to_csv(out / "predictions.csv", index=False)

    thr = THRESHOLD_PP[tgt]
    met = []
    for cfg, g in P.groupby("config"):
        rng = cfg_rng(cfg)  # one fixed stream per method, so a method's interval does not depend on the others
        e = g.err.dropna().to_numpy()
        boot = np.array([np.abs(rng.choice(e, len(e))).mean() for _ in range(N_BOOT)])
        met.append(dict(config=cfg, n=len(e), coverage=g.pred.notna().mean(), mae=np.abs(e).mean(),
                        mae_ci_lo=np.percentile(boot, 2.5), mae_ci_hi=np.percentile(boot, 97.5),
                        rmse=np.sqrt((e ** 2).mean()), bias=e.mean(), within_threshold=(np.abs(e) <= thr).mean()))
    M = pd.DataFrame(met).sort_values("mae")
    M.to_csv(out / "metrics.csv", index=False)

    a = P[P.config == FINAL].set_index("model").err
    comp = []
    for cfg in BASELINES + ABLATIONS:
        rng = cfg_rng("vs_" + cfg)
        b = P[P.config == cfg].set_index("model").err.reindex(a.index)
        ok = b.notna()
        d = (b[ok].abs() - a[ok].abs()).to_numpy()
        boot = np.array([rng.choice(d, len(d)).mean() for _ in range(N_BOOT)])
        comp.append(dict(vs=cfg, n=int(ok.sum()), mae_gap=d.mean(), gap_ci_lo=np.percentile(boot, 2.5),
                         gap_ci_hi=np.percentile(boot, 97.5), wilcoxon_p=wilcoxon(b[ok].abs(), a[ok].abs()).pvalue,
                         final_better=int((d > 0).sum())))
    Cmp = pd.DataFrame(comp)
    Cmp.to_csv(out / "comparisons.csv", index=False)
    print(f"===== {tgt}: test n={len(te)} ({te.release_date.min():%Y-%m-%d} to {te.release_date.max():%Y-%m-%d}), threshold {thr} pp")
    print(M.round(2).to_string(index=False))
    print(Cmp.round(3).to_string(index=False))
