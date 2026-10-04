"""Freeze the temporal test set for each target. Run once; refuses to overwrite.

Test set = models released on or after the date at the (1 - TEST_FRAC) quantile of the target table,
so models sharing a release date always fall on the same side.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C

C.SPLITS.mkdir(parents=True, exist_ok=True)
for tgt in C.TARGETS:
    out = C.SPLITS / f"{tgt}_test_models.txt"
    if out.exists():
        print(f"{out.name} exists; not overwriting (delete manually only with a written reason in notes/)")
        continue
    t = pd.read_csv(C.PROCESSED / f"table_{tgt}.csv", index_col=0, parse_dates=["release_date"])
    cutoff = t.release_date.iloc[int(len(t) * (1 - C.TEST_FRAC))]
    test = t[t.release_date >= cutoff]
    header = (f"# frozen 2026-10-02; target={tgt}; cutoff={cutoff:%Y-%m-%d}; "
              f"test={len(test)} dev={len(t) - len(test)}\n")
    out.write_text(header + "\n".join(test.index) + "\n")
    print(header.strip())
