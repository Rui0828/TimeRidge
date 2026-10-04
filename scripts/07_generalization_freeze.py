"""Generalization check: freeze test splits for every eligible expensive benchmark (rule fixed before any scores
of these targets are examined; see notes/generalization_rule.md). Prints sizes and dates only."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C
from src.data import load_models, load_scores, score_matrix, target_table

b = pd.read_csv(C.RAW_BP / "benchmarks.csv").set_index("benchmark_id")
scores, models = load_scores(), load_models()
W, nc = score_matrix(scores)
W = W.loc[W.index.intersection(models.index)]
nc = nc.loc[W.index]
anyp = W[[p for p in C.PROBES if p in W]].notna().any(axis=1)
OUT = C.SPLITS / "generalization"
OUT.mkdir(parents=True, exist_ok=True)
chosen = []
for c in W.columns:
    cat = str(b.category.get(c, ""))
    cs = json.loads(b.canonical_setting_json.get(c)) if isinstance(b.canonical_setting_json.get(c), str) else {}
    agentic = "Agentic" in cat or c.startswith("swe_") or c.startswith("terminal_bench")
    if not agentic or c in C.PROBES or c in C.TARGETS or "pct" not in cs.get("metric_type", "pct"):
        continue
    if int((W[c].notna() & anyp).sum()) < 15:
        continue
    chosen.append(c)
MIN_DEV = 12  # four-block time-series CV needs at least 4 x 3 development models
for tgt in chosen:
    t = target_table(tgt, W, nc, models, scores)
    cutoff = t.release_date.iloc[int(len(t) * (1 - C.TEST_FRAC))]
    test = t[t.release_date >= cutoff]
    if len(t) - len(test) < MIN_DEV:
        print(f"{tgt}: {len(t) - len(test)} development models, too few for four-fold CV; left out")
        continue
    t.to_csv(C.PROCESSED / f"gen_table_{tgt}.csv")
    f = OUT / f"{tgt}_test_models.txt"
    if f.exists():
        continue
    f.write_text(f"# frozen 2026-10-02; target={tgt}; cutoff={cutoff:%Y-%m-%d}; test={len(test)} dev={len(t) - len(test)}\n"
                 + "\n".join(test.index) + "\n")
    print(f"{tgt:26s} n={len(t):3d} dev={len(t)-len(test):3d} test={len(test):2d} cutoff={cutoff:%Y-%m-%d}")
print(len(chosen), "targets")
