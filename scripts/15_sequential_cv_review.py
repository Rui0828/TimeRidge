"""Recompute only matrix-completion development CV with one validation row per call.

Preserves existing folds, their historical cutoffs, and all supervised predictions.
Outputs are separate from the original results. No test models are predicted.
Run with the /opt/conda base environment and BENCHPRESS_DATA configured.
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

OUT = C.RESULTS / "review" / "sequential_cv"
OUT.mkdir(parents=True, exist_ok=True)
W = pd.read_csv(C.PROCESSED / "score_matrix_verified.csv", index_col=0)
rel = load_models().release_date
W = W.loc[W.index.intersection(rel.index)]
probes = [p for p in C.PROBES if p in W.columns]
original = pd.read_csv(C.RESULTS / "pooled_cv" / "predictions.csv")
rows = []
fold_rows = []

for tgt in sorted(original.target.unique()):
    main = tgt in C.TARGETS
    split = (C.SPLITS / f"{tgt}_test_models.txt" if main else
             C.SPLITS / "generalization" / f"{tgt}_test_models.txt")
    test = {s for s in split.read_text().splitlines() if s and not s.startswith("#")}
    table = C.PROCESSED / (f"table_{tgt}.csv" if main else f"gen_table_{tgt}.csv")
    t = pd.read_csv(table, index_col=0, parse_dates=["release_date"])
    dev = t[~t.index.isin(test)].sort_values(["release_date", "model_name"])
    history = W.drop(index=list(test), errors="ignore")
    size = 6 if main else max(3, round(0.15 * len(dev)))
    n = 0
    for fold in range(4):
        start = len(dev) - (4 - fold) * size
        val = dev.iloc[start:start + size]
        tr = dev[dev.release_date < val.release_date.min()]
        if len(tr) < 4:
            continue
        cutoff = val.release_date.min()
        past = history[rel.loc[history.index] < cutoff]
        assert not set(past.index).intersection(test)
        assert not set(val.index).intersection(test)
        assert (rel.loc[past.index] < cutoff).all()
        fold_rows.append(dict(target=tgt, fold=fold, n_train=len(tr),
                              n_history=len(past), n_validation=len(val), cutoff=cutoff))
        for model in val.index:
            revealed = history.loc[[model], probes]
            assert tgt not in revealed.columns and len(revealed) == 1
            for method, name in [("benchpress", "BenchPress"), ("soft_impute", "Soft-Impute")]:
                prediction = bp.predict(past, revealed, tgt, method=method).iloc[0]
                rows.append(dict(target=tgt, method=name, fold=fold, model=model,
                                 y=val.loc[model, "y"], pred=prediction))
            n += 1
    print(f"{tgt}: {n} development observations, singleton predictions complete", flush=True)

new = pd.DataFrame(rows)
new["err"] = new.pred - new.y
old_matrix = original[original.method.isin(["BenchPress", "Soft-Impute"])]
key = ["target", "method", "model"]
assert not new.duplicated(key).any()
assert set(map(tuple, new[key].to_numpy())) == set(map(tuple, old_matrix[key].to_numpy()))
assert np.allclose(new.set_index(key).y.sort_index(), old_matrix.set_index(key).y.sort_index())
new.to_csv(OUT / "matrix_predictions.csv", index=False)
pd.DataFrame(fold_rows).to_csv(OUT / "folds.csv", index=False)

# The untouched supervised predictions remain byte-for-byte equivalent as data.
P = pd.concat([original[~original.method.isin(["BenchPress", "Soft-Impute"])],
               new[original.columns]], ignore_index=True)
assert P.groupby("method").size().eq(310).all()
assert not P.duplicated(key).any()
P.to_csv(OUT / "predictions.csv", index=False)
per = P.groupby(["target", "method"]).agg(
    n=("err", "size"), bias=("err", "mean"), mae=("err", lambda e: e.abs().mean())).reset_index()
per.to_csv(OUT / "per_target.csv", index=False)
reference = P[P.method.eq("TimeRidge")].set_index(["target", "model"]).err.abs()
summary = []
for method, group in P.groupby("method"):
    b = per[per.method.eq(method)]
    ae = group.set_index(["target", "model"]).err.abs().reindex(reference.index)
    under = int(b.bias.lt(0).sum())
    summary.append(dict(method=method, benchmarks=len(b), under=under,
                        sign_test_p=binomtest(under, len(b), 0.5, alternative="greater").pvalue,
                        n_predictions=len(group), pooled_bias=group.err.mean(),
                        bias_p_two_sided=ttest_1samp(group.err, 0).pvalue,
                        pooled_mae=group.err.abs().mean(), pooled_rmse=np.sqrt((group.err ** 2).mean()),
                        wilcoxon_p_vs_timeridge=np.nan if method == "TimeRidge" else
                        wilcoxon(reference, ae).pvalue))
summary = pd.DataFrame(summary).sort_values("pooled_mae")
summary.to_csv(OUT / "pooled.csv", index=False)
blocks = new.groupby(["target", "method", "fold"]).agg(
    n=("err", "size"), mae=("err", lambda e: e.abs().mean()), bias=("err", "mean")).reset_index()
blocks.to_csv(OUT / "matrix_block_metrics.csv", index=False)
comparison = old_matrix.merge(new, on=key, suffixes=("_batch", "_singleton"))
comparison["prediction_change_pp"] = comparison.pred_singleton - comparison.pred_batch
comparison.to_csv(OUT / "batch_vs_singleton.csv", index=False)
print(summary.round(4).to_string(index=False), flush=True)
print(f"Saved corrected development CV to {OUT}; original and test outputs preserved.", flush=True)
