"""Supplementary §7 numbers from existing pooled predictions only (no model is fit, nothing is tuned).

  1. CV MAE and bias as mean ± std over the four expanding-window blocks (each block pooled over the 20 benchmarks).
  2. Share of predictions within the decision threshold where §6 defines one (Terminal-Bench 2.0: 5.3 points;
     SWE-Bench Pro: 1.8, as in scripts/05_cv.py). BenchPress's num_problems counts generations for some benchmarks
     (Terminal-Bench 2.0: 89 tasks x 5 = 445) and is missing for others, so no threshold is derived for the other 18.
  3. Test-set slices: open vs closed weights, number of inputs reported, benchmark family.
  4. Benjamini-Hochberg correction over every pooled test (t-test, sign test, Wilcoxon) in CV and test.
Output: results/results_stats/*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C

OUT = C.RESULTS / "results_stats"
OUT.mkdir(parents=True, exist_ok=True)
SHOW = ["Training mean", "OLS on GPQA", "GBM", "BenchPress", "Soft-Impute", "TimeRidge",
        "TimeRidge - time", "Set A (raw)", "Set B (top-k)", "Set C, all shrunk"]


def tables():
    """(target, full table, test model list) for all 20 benchmarks."""
    for tgt in C.TARGETS:
        t = pd.read_csv(C.PROCESSED / f"table_{tgt}.csv", index_col=0, parse_dates=["release_date"])
        yield tgt, t, [l for l in (C.SPLITS / f"{tgt}_test_models.txt").read_text().splitlines() if l and not l.startswith("#")]
    for f in sorted((C.SPLITS / "generalization").glob("*_test_models.txt")):
        tgt = f.name.removesuffix("_test_models.txt")
        t = pd.read_csv(C.PROCESSED / f"gen_table_{tgt}.csv", index_col=0, parse_dates=["release_date"])
        yield tgt, t, [l for l in f.read_text().splitlines() if l and not l.startswith("#")]


# ---------- block index of each CV prediction ----------
fold_rows, meta = [], []
for tgt, t, test in tables():
    dev = t[~t.index.isin(test)].sort_values(["release_date", "model_name"])
    if tgt in C.TARGETS:  # 05_cv.py records the block
        cvp = pd.read_csv(C.RESULTS / "cv" / tgt / "predictions.csv")
        fm = cvp.drop_duplicates("model").set_index("model").fold
    else:  # same rule as 11_pooled_cv.py: four blocks of max(3, 15% of dev) at the end of the dev split
        size = max(3, round(0.15 * len(dev)))
        fm = pd.Series({m: k for k in range(4) for m in dev.index[len(dev) - (4 - k) * size: len(dev) - (3 - k) * size]})
    fold_rows += [dict(target=tgt, model=m, fold=int(k)) for m, k in fm.items()]
    P = [c for c in C.PROBES if c in t]
    meta.append(pd.DataFrame({"target": tgt, "model": t.index, "open_weights": t.open_weights.astype(bool).values,
                              "n_inputs": t[P].notna().sum(1).values}))
folds = pd.DataFrame(fold_rows)
meta = pd.concat(meta)

# ---------- 1. CV mean ± std over blocks ----------
cv = pd.read_csv(C.RESULTS / "pooled_cv" / "predictions.csv").merge(folds, on=["target", "model"], how="left")
assert cv.fold.notna().all(), "unmatched CV prediction"
cv["err"] = cv.pred - cv.y
blk = cv.groupby(["method", "fold"]).agg(mae=("err", lambda e: e.abs().mean()), bias=("err", "mean")).reset_index()
S1 = blk.groupby("method").agg(mae_mean=("mae", "mean"), mae_std=("mae", "std"),
                               bias_mean=("bias", "mean"), bias_std=("bias", "std"), n_blocks=("fold", "nunique"))
S1.to_csv(OUT / "cv_fold_mean_std.csv")
print("CV over the four blocks (each pooled over 20 benchmarks):")
print(S1.loc[[m for m in SHOW if m in S1.index]].round(2).to_string())

# ---------- 2. within-threshold shares (benchmarks with a defined threshold) ----------
THR = {"terminal_bench": 5.3, "swe_bench_pro": 1.8}
te = pd.read_csv(C.RESULTS / "pooled_test" / "predictions.csv")
rows = []
for split, D in [("cv", cv), ("test", te)]:
    for (tgt, m), g in D[D.target.isin(list(THR))].groupby(["target", "method"]):
        rows.append(dict(target=tgt, split=split, method=m, n=len(g), within=(g.err.abs() <= THR[tgt]).mean(),
                         mae=g.err.abs().mean()))
S2 = pd.DataFrame(rows)
S2.to_csv(OUT / "within_threshold.csv", index=False)
print("\nshare within the decision threshold (TB 5.3, SWE-Pro 1.8 points):")
print(S2[S2.method.isin(SHOW)].pivot_table(index="method", columns=["target", "split"], values="within").loc[SHOW].round(2).to_string())

# ---------- 3. test slices ----------
te = te.merge(meta, on=["target", "model"], how="left")
fam = lambda t: "SWE-bench" if t.startswith("swe_bench") else ("Terminal-Bench" if t.startswith("terminal_bench") else "other agentic")
te["family"] = te.target.map(fam)
te["inputs"] = np.where(te.n_inputs <= 5, "<=5 inputs", ">5 inputs")
te["weights"] = np.where(te.open_weights, "open", "closed")
rows = []
for col in ["weights", "inputs", "family"]:
    for (lvl, m), g in te.groupby([col, "method"]):
        if m in ["TimeRidge", "TimeRidge - time", "Set B (top-k)", "GBM", "BenchPress"]:
            rows.append(dict(slice=col, level=lvl, method=m, n=len(g), bias=g.err.mean(), mae=g.err.abs().mean()))
S3 = pd.DataFrame(rows)
S3.to_csv(OUT / "test_slices.csv", index=False)
print("\ntest slices:")
print(S3.pivot_table(index=["slice", "level"], columns="method", values=["mae", "bias"]).round(1).to_string())
print(te[te.method == "TimeRidge"].groupby(["weights", "inputs", "family"]).size().to_string())

# ---------- 4. multiple comparisons: Benjamini-Hochberg over every pooled test reported in §7 ----------
from scipy.stats import false_discovery_control

rows = []
for split in ("cv", "test"):
    D = pd.read_csv(C.RESULTS / f"pooled_{split}" / "pooled.csv")
    for _, r in D.iterrows():
        rows.append(dict(split=split, test="t-test bias", method=r.method, p=r.bias_p_two_sided))
        if r.method not in ("TimeRidge", "TimeRidge - source", "TimeRidge - missing ind."):
            rows.append(dict(split=split, test="sign test", method=r.method, p=r.sign_test_p))
        if pd.notna(r.wilcoxon_p_vs_timeridge):
            rows.append(dict(split=split, test="Wilcoxon vs TimeRidge", method=r.method, p=r.wilcoxon_p_vs_timeridge))
S4 = pd.DataFrame(rows)
S4["p_bh"] = false_discovery_control(S4.p.to_numpy())
S4["verdict_changes"] = (S4.p < 0.05) != (S4.p_bh < 0.05)
S4.to_csv(OUT / "multiple_comparisons.csv", index=False)
print(f"\nBenjamini-Hochberg over {len(S4)} pooled tests: {int(S4.verdict_changes.sum())} verdicts change at 0.05")
