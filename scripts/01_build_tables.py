"""Build cleaned per-target tables and the full score matrix into data/processed/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C
from src.data import load_models, load_scores, score_matrix, target_table

C.PROCESSED.mkdir(parents=True, exist_ok=True)
scores = load_scores()
models = load_models()
W, nc = score_matrix(scores)
W = W.loc[W.index.intersection(models.index)]
nc = nc.loc[W.index]
W.to_csv(C.PROCESSED / "score_matrix_verified.csv")
nc.to_csv(C.PROCESSED / "noncanonical_flags.csv")
print(f"score matrix: {W.shape[0]} models x {W.shape[1]} benchmarks, {int(W.notna().sum().sum())} cells")

for tgt in C.TARGETS:
    t = target_table(tgt, W, nc, models, scores)
    t.to_csv(C.PROCESSED / f"table_{tgt}.csv")
    cov = {p: int(t[p].notna().sum()) for p in C.PROBES if p in t}
    miss = {k: v for k, v in sorted(cov.items(), key=lambda kv: -kv[1])[:6]}  # top-6 coverage
    print(f"{tgt}: {len(t)} models, {t.release_date.min():%Y-%m-%d} to {t.release_date.max():%Y-%m-%d}; "
          f"target non-canonical {int(t.target_noncanonical.sum())}; third-party {int(t.target_third_party.sum())}; probe coverage (top 6) {miss}; "
          f"reasoning {int(t.is_reasoning.eq(True).sum())}/{len(t)}")
