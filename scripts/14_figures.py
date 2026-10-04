"""Paper figures from saved results (no model fitting here). Figure 1 is drawn by scripts/03_efa.py.

fig_protocol:       release-order evaluation on the real Terminal-Bench models (Figure 2)
fig_generalization: test-set signed bias on all 20 benchmarks, with and without the time term (Figure 3)
fig_coef_full:      all standardized coefficients of the final Terminal-Bench model, bootstrap CI, p, VIF (Figure 4)
Output: results/figures/*.{pdf,png}
"""
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C

OUT = C.RESULTS / "figures"
OUT.mkdir(parents=True, exist_ok=True)
INK2, MUTED, GRID, AXIS, SURF = "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
mpl.rcParams.update({"font.size": 8, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.5, "figure.dpi": 200, "savefig.bbox": "tight",
                     "font.family": "DejaVu Sans"})
tgt = C.MAIN_TARGET


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)


def nice(f):
    if f == "t_years": return "release time"
    if f == "third_party": return "third-party source"
    short = {"aider_polyglot_diff": "Aider (diff)", "aider_polyglot_whole": "Aider (whole)",
             "bullshit_pushback": "Bullshit-Bench", "tau2_bench_retail": "τ²-Retail", "tau2_bench_telecom": "τ²-Telecom",
             "tau2_bench_airline": "τ²-Airline", "gpqa_diamond": "GPQA Diamond", "mmlu_pro": "MMLU-Pro",
             "aime_2025": "AIME 2025", "aime_2026": "AIME 2026", "arena_hard": "Arena-Hard", "aa_lcr": "AA-LCR",
             "hmmt_feb_2025": "HMMT Feb 2025", "hmmt_feb_2026": "HMMT Feb 2026", "hmmt_nov_2025": "HMMT Nov 2025",
             "matharena_apex_2025": "MathArena Apex", "math_500": "MATH-500", "scicode": "SciCode", "critpt": "CritPt",
             "ifbench": "IFBench", "brumo_2025": "BRUMO 2025", "smt_2025": "SMT 2025", "cmimc_2025": "CMIMC 2025",
             "vibe_eval": "Vibe-Eval", "alpacaeval_2": "AlpacaEval 2"}
    k = f.removeprefix("logit_").removeprefix("missing_")
    name = short.get(k, k)
    return ("missing " if f.startswith("missing_") else "") + name


# ---- fig_protocol (§6): release-order evaluation on the real Terminal-Bench models
T = pd.read_csv(C.PROCESSED / f"table_{tgt}.csv", index_col=0, parse_dates=["release_date"])
tst = {l for l in (C.SPLITS / f"{tgt}_test_models.txt").read_text().splitlines() if l and not l.startswith("#")}
dv = T[~T.index.isin(tst)].sort_values(["release_date", "model_name"])
ts = T[T.index.isin(tst)].sort_values("release_date")
NF, VS = 4, 6
blocks = [dv.iloc[len(dv) - (NF - f) * VS: len(dv) - (NF - f) * VS + VS] for f in range(NF)]
in_val = pd.concat(blocks).index
GRAY = "#B4B2A9"
with mpl.rc_context({"font.size": 7}):
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(3.03, 2.75), sharex=True, gridspec_kw={"height_ratios": [2.1, 1.25], "hspace": 0.08})
    cut = ts.release_date.min()
    for a_ in (ax, bx):
        a_.axvline(cut, color="#2c2c2a", lw=1, ls=(0, (3, 2)), zorder=1)
    tr_only = dv[~dv.index.isin(in_val)]
    ax.scatter(tr_only.release_date, tr_only.y, s=12, color=GRAY, edgecolor=SURF, lw=0.5, zorder=3, label="training only")
    vv = dv.loc[in_val]
    ax.scatter(vv.release_date, vv.y, s=12, color=BLUE, edgecolor=SURF, lw=0.5, zorder=3, label="CV validation")
    ax.scatter(ts.release_date, ts.y, s=14, color=ORANGE, edgecolor=SURF, lw=0.5, zorder=3, label="test")
    ax.set_ylabel("Terminal-Bench 2.0\nscore (%)")
    ax.text(cut, 102, " test cutoff", fontsize=6.3, color=INK2, va="top")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=6.5, handletextpad=0.2, columnspacing=0.8)
    t0 = dv.release_date.min()
    rows = [(f"CV {f + 1}", b_) for f, b_ in enumerate(blocks)]
    for r, (lab, b_) in enumerate(rows):
        yy = len(rows) - r
        bx.plot([t0, b_.release_date.min()], [yy, yy], color=GRAY, lw=3, solid_capstyle="butt")
        bx.plot([b_.release_date.min(), b_.release_date.max()], [yy, yy], color=BLUE, lw=3, solid_capstyle="butt")
    bx.plot([t0, cut], [0, 0], color=GRAY, lw=3, solid_capstyle="butt")
    bx.plot([cut, ts.release_date.max()], [0, 0], color=ORANGE, lw=3, solid_capstyle="butt")
    bx.set_yticks(range(len(rows) + 1), ["test"] + [lab for lab, _ in rows][::-1])
    bx.set_ylim(-0.7, len(rows) + 0.7)
    bx.grid(axis="y", visible=False)
    bx.xaxis.set_major_locator(mpl.dates.MonthLocator(bymonth=(1, 4, 7, 10)))
    bx.xaxis.set_major_formatter(mpl.dates.DateFormatter("%b\n%Y"))
    bx.set_xlabel("release date")
    plt.setp(bx.get_xticklabels(), rotation=0)
    save(fig, "fig_protocol")

# ---- fig_generalization: with vs without the time term (same ridge), BenchPress as reference
G = pd.read_csv(C.RESULTS / "pooled_test" / "per_target.csv")
G = G[G.method.isin(["TimeRidge", "TimeRidge - time", "BenchPress"])].replace(
    {"method": {"TimeRidge": "Final model", "TimeRidge - time": "Final model - time"}})
pv = G.pivot(index="target", columns="method", values="bias")
nn = G.groupby("target").n.first()
order = pv.sort_values("Final model - time").index
bnames = pd.read_csv(C.RAW_BP / "benchmarks.csv").set_index("benchmark_id").benchmark_name
SHORT = {"tau_bench_airline": "τ-bench Airline", "tau_bench_retail": "τ-bench Retail",
         "toolathlon": "Toolathlon", "claw_eval_pass3": "Claw Eval (pass³)", "mcpatlas": "MCPAtlas",
         "terminal_bench": "Terminal-Bench 2.0", "swe_bench_pro": "SWE-Bench Pro"}
with mpl.rc_context({"font.size": 7}):
    fig, ax = plt.subplots(figsize=(3.03, 2.95))
    ax.axvspan(-100, 0, color="#898781", alpha=0.07, lw=0, zorder=0)
    ax.axvline(0, color="#2c2c2a", lw=1.4, zorder=2)
    y = np.arange(len(order))
    for yi, t_ in zip(y, order):
        a, b = pv.loc[t_, "Final model - time"], pv.loc[t_, "Final model"]
        ax.annotate("", xy=(b, yi), xytext=(a, yi), zorder=3,
                    arrowprops=dict(arrowstyle="-|>", color="#B4B2A9", lw=1.0, shrinkA=2.5, shrinkB=3, mutation_scale=6))
    h1 = ax.scatter(pv.loc[order, "Final model - time"], y, s=16, color="#898781", edgecolor=SURF, lw=0.6, zorder=4)
    h2 = ax.scatter(pv.loc[order, "Final model"], y, s=16, color=ORANGE, edgecolor=SURF, lw=0.6, zorder=4)
    h3 = ax.scatter(pv.loc[order, "BenchPress"], y, s=34, marker="|", color=BLUE, alpha=0.85, linewidths=1.6, zorder=1.8)
    ax.set_yticks(y, [f"{SHORT.get(t_, str(bnames.get(t_, t_)))} ({nn[t_]})" for t_ in order])
    ax.set_ylim(-0.8, len(order) + 0.5)
    ax.set_xlabel("test signed bias, predicted − actual (pp)")
    ax.set_xlim(-45, 25)
    ax.text(-1.5, len(order) - 0.05, "← underpredicts", fontsize=6.3, color=INK2, va="center", ha="right")
    ax.text(1.5, len(order) - 0.05, "overpredicts →", fontsize=6.3, color=INK2, va="center", ha="left")
    ax.legend([h1, h2, h3], ["TimeRidge − time", "TimeRidge (ours)", "BenchPress"], loc="lower center",
              bbox_to_anchor=(0.4, 1.0), ncol=3, frameon=False, fontsize=6.5, handletextpad=0.2, columnspacing=0.8)
    save(fig, "fig_generalization")


# ---- fig_coef_full (main target): every feature of the final model, two panels, bootstrap CI, p, VIF (core)
R = pd.read_csv(C.RESULTS / "coefficients" / f"{tgt}.csv").set_index("feature")
V = pd.read_csv(C.RESULTS / "coefficients" / f"{tgt}_vif.csv", index_col=0).iloc[:, 0]
R = R.reindex(R.coef_per_sd.abs().sort_values(ascending=False).index)
half = (len(R) + 1) // 2
with mpl.rc_context({"font.size": 6}):
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.75))
    for ax, part in zip(axes, [R.iloc[:half], R.iloc[half:]]):
        part = part.iloc[::-1]
        ax.axvline(0, color=AXIS, lw=0.8)
        for i, (f, r) in enumerate(part.iterrows()):
            sig = r.ci_lo > 0 or r.ci_hi < 0
            col = ORANGE if sig else BLUE
            ax.plot([r.ci_lo, r.ci_hi], [i, i], color=col, lw=1.4, solid_capstyle="round")
            ax.scatter(r.coef_per_sd, i, s=8, color=col, zorder=3)
            pv = "<.001" if r.p_bootstrap < 0.001 else f"{r.p_bootstrap:.3f}"
            vf = f"{V[f]:.1f}" if f in V.index else "–"
            for txt, dx in ((pv, 5), (vf, 30)):
                ax.annotate(txt, xy=(1.0, i), xycoords=("axes fraction", "data"), xytext=(dx, 0),
                            textcoords="offset points", va="center", fontsize=5.6, color=INK2)
        for txt, dx in (("p", 5), ("VIF", 30)):
            ax.annotate(txt, xy=(1.0, len(part) - 0.3), xycoords=("axes fraction", "data"), xytext=(dx, 0),
                        textcoords="offset points", va="bottom", fontsize=5.6, color=INK2)
        ax.set_yticks(range(len(part)), [nice(f) for f in part.index])
        ax.set_ylim(-0.7, len(part) - 0.3)
        ax.grid(axis="y", visible=False)
    axes[0].set_xlabel("standardized coefficient (logit per 1 SD)")
    axes[1].set_xlabel("standardized coefficient (logit per 1 SD)")
    fig.tight_layout(w_pad=6.5)
    save(fig, "fig_coef_full")
