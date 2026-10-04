"""Exploratory feature analysis on the DEV set only (test models excluded).

Outputs (results/efa/):
  summary_stats.csv       §3.2 table
  correlations.csv        Pearson / Spearman of logit(target) with each feature
  probe_redundancy.csv    pairwise probe correlations and VIF (complete cases)
  hypotheses.csv          H1-H3 tests
  fig_scatter.{pdf,png}   target vs GPQA / MMLU-Pro, colored by release date, older vs newer half fits
  fig_residual_time.{pdf,png}  residual of logit(target) ~ logit(GPQA) vs release date
"""
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C
from src.data import inv_logit, logit

OUT = C.RESULTS / "efa"
OUT.mkdir(parents=True, exist_ok=True)

# palette (dataviz reference palette, light mode)
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
mpl.rcParams.update({"font.size": 8, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "figure.dpi": 200,
                     "savefig.bbox": "tight", "font.family": "DejaVu Sans"})

NAMES = {"gpqa_diamond": "GPQA Diamond", "mmlu_pro": "MMLU-Pro", "aime_2025": "AIME 2025",
         "livecodebench": "LiveCodeBench", "swe_bench_pro": "SWE-Bench Pro", "terminal_bench": "Terminal-Bench 2.0"}


def dev_table(tgt):
    t = pd.read_csv(C.PROCESSED / f"table_{tgt}.csv", index_col=0, parse_dates=["release_date"])
    test = {l for l in (C.SPLITS / f"{tgt}_test_models.txt").read_text().splitlines() if l and not l.startswith("#")}
    return t[~t.index.isin(test)].copy()


summ, corr, red, hyp = [], [], [], []
for tgt in C.TARGETS:
    d = dev_table(tgt)
    d["ly"] = logit(d.y)
    for p in C.PROBES:
        d["l_" + p] = logit(d[p])
    ys = d.y.sort_values().to_numpy()
    summ.append(dict(target=tgt, n=len(d), date_min=f"{d.release_date.min():%Y-%m-%d}",
                     date_max=f"{d.release_date.max():%Y-%m-%d}", y_mean=d.y.mean(), y_sd=d.y.std(),
                     y_min=d.y.min(), y_max=d.y.max(), n_providers=d.provider.nunique(),
                     target_noncanonical=int(d.target_noncanonical.sum()),
                     **{f"missing_{p}": int(d[p].isna().sum()) for p in C.PROBES},
                     **{f"mean_{p}": d[p].mean() for p in C.PROBES},
                     decision_threshold_median_adjacent_gap=float(np.median(np.diff(ys)))))
    for f in ["l_" + p for p in C.PROBES if d[p].notna().sum() >= 8] + ["t_years"]:
        x = d[[f, "ly"]].dropna()
        corr.append(dict(target=tgt, feature=f, n=len(x), pearson=pearsonr(x[f], x.ly)[0],
                         pearson_p=pearsonr(x[f], x.ly)[1], spearman=spearmanr(x[f], x.ly)[0]))
    # redundancy among the two well-covered probes + time (complete cases)
    cc = d[["l_gpqa_diamond", "l_mmlu_pro", "t_years"]].dropna()
    X = sm.add_constant(cc)
    for i, c in enumerate(cc.columns, start=1):
        red.append(dict(target=tgt, feature=c, n=len(cc), vif=variance_inflation_factor(X.values, i),
                        r_with_gpqa=cc[c].corr(cc.l_gpqa_diamond)))
    # H1: drift — at the same GPQA, newer models score higher on the target
    r1 = smf.ols("ly ~ l_gpqa_diamond + t_years", d).fit(cov_type="HC3")
    # H2: MMLU-Pro adds information beyond GPQA
    r2 = smf.ols("ly ~ l_gpqa_diamond + l_mmlu_pro", d.dropna(subset=["l_mmlu_pro"])).fit(cov_type="HC3")
    # H3: non-canonical (vendor-scaffold) target cells sit above the GPQA line
    r3 = smf.ols("ly ~ l_gpqa_diamond + target_noncanonical", d.assign(
        target_noncanonical=d.target_noncanonical.astype(int))).fit(cov_type="HC3")
    for h, r, term in [("H1 drift", r1, "t_years"), ("H2 MMLU-Pro beyond GPQA", r2, "l_mmlu_pro"),
                       ("H3 non-canonical above line", r3, "target_noncanonical")]:
        lo, hi = r.conf_int().loc[term]
        hyp.append(dict(target=tgt, hypothesis=h, term=term, n=int(r.nobs), coef=r.params[term], ci_lo=lo, ci_hi=hi,
                        p_HC3=r.pvalues[term], r2=r.rsquared))

    if tgt != C.MAIN_TARGET:
        continue
    # ---- figures (main target only) ----
    tnum = mpl.dates.date2num(d.release_date)
    norm = mpl.colors.Normalize(tnum.min(), tnum.max())
    half = d.release_date.iloc[len(d) // 2]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.6), sharey=True)
    for ax, p in zip(axes, ["gpqa_diamond", "mmlu_pro"]):
        dd = d.dropna(subset=[p])
        sc = ax.scatter(dd[p], dd.y, c=mpl.dates.date2num(dd.release_date), cmap=SEQ, norm=norm, s=22,
                        edgecolor="#fcfcfb", linewidth=0.8, zorder=3)
        for part, col, lab in [(dd[dd.release_date < half], BLUE, "older half fit"),
                               (dd[dd.release_date >= half], ORANGE, "newer half fit")]:
            if len(part) < 5:
                continue
            part = part.dropna(subset=[p])
            b = np.polyfit(logit(part[p]), logit(part.y), 1)
            xs = np.linspace(dd[p].min(), dd[p].max(), 100)
            ax.plot(xs, inv_logit(np.polyval(b, logit(xs))), color=col, lw=2, zorder=2, label=lab)
        r = pearsonr(logit(dd[p]), logit(dd.y))[0]
        ax.set_xlabel(f"{NAMES[p]} score (%)")
        ax.text(0.03, 0.95, f"r(logit) = {r:+.2f}, n = {len(dd)}", transform=ax.transAxes, va="top", color=INK2)
    axes[0].set_ylabel(f"{NAMES[tgt]} score (%)")
    axes[0].legend(loc="lower right", frameon=False, fontsize=7)
    cb = fig.colorbar(sc, ax=axes, pad=0.02, fraction=0.04)
    cb.ax.yaxis.set_major_formatter(mpl.dates.DateFormatter("%Y-%m"))
    cb.set_label("release date", color=INK2)
    cb.outline.set_visible(False)
    for ext in ["pdf", "png"]:
        fig.savefig(OUT / f"fig_scatter.{ext}")
    plt.close(fig)

    # single-column print-size version for the paper (GPQA panel only)
    with mpl.rc_context({"font.size": 7}):
        fig, ax = plt.subplots(figsize=(3.03, 1.95))
        p = "gpqa_diamond"
        sc = ax.scatter(d[p], d.y, c=mpl.dates.date2num(d.release_date), cmap=SEQ, norm=norm, s=16,
                        edgecolor="#fcfcfb", linewidth=0.6, zorder=3)
        for part, col, lab in [(d[d.release_date < half], BLUE, "older half"),
                               (d[d.release_date >= half], ORANGE, "newer half")]:
            part = part.dropna(subset=[p])
            b = np.polyfit(logit(part[p]), logit(part.y), 1)
            xs = np.linspace(d[p].min(), d[p].max(), 100)
            ax.plot(xs, inv_logit(np.polyval(b, logit(xs))), color=col, lw=1.8, zorder=2, label=f"fit, {lab}")
        ax.set_xlabel("GPQA Diamond score (%)")
        ax.set_ylabel(f"{NAMES[tgt]} score (%)")
        ax.legend(loc="upper left", frameon=False)
        cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.05)
        cb.ax.yaxis.set_major_formatter(mpl.dates.DateFormatter("%Y-%m"))
        cb.set_label("release date", color=INK2)
        cb.outline.set_visible(False)
        for ext in ["pdf", "png"]:
            fig.savefig(OUT / f"fig_scatter_col.{ext}")
        plt.close(fig)

    d = d.dropna(subset=["l_gpqa_diamond"])  # residual figure needs GPQA
    base = smf.ols("ly ~ l_gpqa_diamond", d).fit()
    fig, ax = plt.subplots(figsize=(3.1, 2.4))
    resid_pp = d.y - inv_logit(base.fittedvalues)
    ax.axhline(0, color=AXIS, lw=1, zorder=1)
    ax.scatter(d.release_date, resid_pp, s=22, color=BLUE, edgecolor="#fcfcfb", linewidth=0.8, zorder=3)
    b = np.polyfit(d.t_years, resid_pp, 1)
    xs = np.linspace(d.t_years.min(), d.t_years.max(), 50)
    ax.plot(pd.Timestamp("2025-01-01") + pd.to_timedelta(xs * 365.25, unit="D"), np.polyval(b, xs),
            color=ORANGE, lw=2, zorder=2)
    ax.text(0.03, 0.95, f"trend {b[0]:+.1f} pp / year", transform=ax.transAxes, va="top", color=INK2)
    ax.set_xlabel("release date")
    ax.set_ylabel("residual, actual − fit (pp)")
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mpl.dates.MonthLocator(bymonth=[1, 4, 7, 10]))
    for ext in ["pdf", "png"]:
        fig.savefig(OUT / f"fig_residual_time.{ext}")
    plt.close(fig)

for name, rows in [("summary_stats", summ), ("correlations", corr), ("probe_redundancy", red), ("hypotheses", hyp)]:
    pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False)
    print(f"--- {name}\n{pd.DataFrame(rows).round(3).to_string(index=False)}")
