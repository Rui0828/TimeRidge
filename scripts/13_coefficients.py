"""§8 coefficients of the final model (Set C, time and source unpenalized), fit on the DEV split of each target.

With ~50 features and ~41 dev models OLS inference is not defined (p > n), so we report standardized Ridge
coefficients with 2,000-resample bootstrap 95% CIs and a two-sided bootstrap p-value (sign-flip share).
VIF is computed on the core features (time, source, and probes observed for at least half of the dev models).
Output: results/coefficients/<target>.csv (all features) and <target>_vif.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C
from src.data import logit
from src.models import Linear, features

OUT = C.RESULTS / "coefficients"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(C.SEED)
FREE_SCALE = 1e3

for tgt in C.TARGETS:
    t = pd.read_csv(C.PROCESSED / f"table_{tgt}.csv", index_col=0, parse_dates=["release_date"])
    test = {l for l in (C.SPLITS / f"{tgt}_test_models.txt").read_text().splitlines() if l and not l.startswith("#")}
    dev = t[~t.index.isin(test)]
    alpha = pd.read_csv(C.RESULTS / "cv" / tgt / "summary.csv").set_index("config").loc["ridge_C_free_controls", "alpha"]
    X = features(dev, "C")
    X = X.loc[:, X.notna().any()]
    X = X.fillna(X.mean())
    X = X.loc[:, X.std() > 0]
    y = pd.Series(logit(dev.y), index=dev.index)
    Z = pd.DataFrame(StandardScaler().fit_transform(X), index=X.index, columns=X.columns)
    scale = np.array([FREE_SCALE if c in Linear.FREE else 1.0 for c in Z.columns])

    def fit(Zs, ys):
        return Ridge(alpha=alpha).fit(Zs * scale, ys).coef_ * scale  # coefficients per 1 SD

    coef = fit(Z.to_numpy(), y.to_numpy())
    boot = np.array([fit(Z.to_numpy()[i], y.to_numpy()[i])
                     for i in (rng.integers(0, len(Z), len(Z)) for _ in range(2000))])
    p_boot = 2 * np.minimum((boot > 0).mean(0), (boot < 0).mean(0))
    res = pd.DataFrame({"coef_per_sd": coef, "ci_lo": np.percentile(boot, 2.5, 0), "ci_hi": np.percentile(boot, 97.5, 0),
                        "p_bootstrap": np.clip(p_boot, 1 / 2000, 1), "sd_raw": X.std().values,
                        "coverage_dev": [dev[c.removeprefix("logit_")].notna().mean() if c.startswith("logit_") else np.nan
                                         for c in Z.columns]}, index=Z.columns)
    res.index.name = "feature"
    res.sort_values("coef_per_sd", key=np.abs, ascending=False).to_csv(OUT / f"{tgt}.csv")

    core = ["t_years", "third_party"] + [f"logit_{p}" for p in C.PROBES
                                          if f"logit_{p}" in Z and dev[p].notna().mean() >= 0.5]
    Zc = sm.add_constant(Z[core]).to_numpy()
    vif = pd.Series([variance_inflation_factor(Zc, j + 1) for j in range(len(core))], index=core, name="vif")
    vif.to_csv(OUT / f"{tgt}_vif.csv")

    p = dev.y.mean() / 100
    per_year = res.loc["t_years", "coef_per_sd"] / res.loc["t_years", "sd_raw"]
    print(f"===== {tgt}: dev n={len(dev)}, features={Z.shape[1]}, alpha={alpha:.3g}")
    print(res.sort_values("coef_per_sd", key=np.abs, ascending=False).head(12).round(3).to_string())
    print(vif.round(2).to_string())
    print(f"time: {per_year:+.2f} logit/yr = {per_year * p * (1 - p) * 100:+.1f} pp/yr at dev mean {p*100:.1f}%")
