"""EFA follow-up (dev set only): does the time drift survive controlling for the score source?

logit(target) ~ logit(GPQA) + release time [+ third-party-source dummy], HC3, under four specs:
all dev models; + third-party dummy; vendor-reported only; vendor-reported and canonical only.
Output: results/efa/source_check.csv
"""
import sys
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C
from src.data import logit

rows = []
for tgt in C.TARGETS:
    t = pd.read_csv(C.PROCESSED / f"table_{tgt}.csv", index_col=0, parse_dates=["release_date"])
    test = {l for l in (C.SPLITS / f"{tgt}_test_models.txt").read_text().splitlines() if l and not l.startswith("#")}
    d = t[~t.index.isin(test)].copy()
    d["ly"], d["lg"], d["third"] = logit(d.y), logit(d.gpqa_diamond), d.target_third_party.astype(int)
    specs = [("all", "ly ~ lg + t_years", d),
             ("+ third-party dummy", "ly ~ lg + t_years + third", d),
             ("vendor-reported only", "ly ~ lg + t_years", d[d.third == 0]),
             ("vendor-reported & canonical", "ly ~ lg + t_years", d[(d.third == 0) & ~d.target_noncanonical])]
    for name, f, dd in specs:
        r = smf.ols(f, dd).fit(cov_type="HC3")
        lo, hi = r.conf_int().loc["t_years"]
        rows.append(dict(target=tgt, spec=name, n=int(r.nobs), time_coef=r.params.t_years, ci_lo=lo, ci_hi=hi,
                         p=r.pvalues.t_years, third_coef=r.params.get("third"), third_p=r.pvalues.get("third")))
    rows.append(dict(target=tgt, spec="source hosts", n=len(d),
                     time_coef=None, third_coef=None, p=None, ci_lo=None, ci_hi=None, third_p=None,
                     **{"hosts": "; ".join(f"{h}:{n}" for h, n in d.target_source_host.value_counts().items())}))
out = pd.DataFrame(rows)
out.to_csv(C.RESULTS / "efa" / "source_check.csv", index=False)
print(out.drop(columns=["hosts"]).round(4).to_string(index=False))
