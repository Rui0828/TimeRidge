"""Feature sets and per-target predictors. Every fitted quantity (imputation means, scaling,
top-k selection, weights) is computed from the training rows passed to `fit` only.

Feature sets (HW slide 15). Probes = the 25-benchmark BenchPress low-cost allowlist (config.PROBES).
  A  raw probe scores (%), mean-imputed, + open_weights
  B  top-k of Set A columns by |Pearson r| with the target in the training rows
  C  logit probes + a missing indicator per probe + release time + third-party-source control
     (predictions are made with third_party = 0, i.e. the vendor-reported score)
A probe never observed in a training set is dropped for that fit.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from src import config as C
from src.data import inv_logit, logit


def _impute(X_tr, X_te):
    mu = X_tr.mean()
    return X_tr.fillna(mu), X_te.fillna(mu)


def features(df: pd.DataFrame, fset: str, probes=None) -> pd.DataFrame:
    """Raw (unfitted) feature columns; fitting-dependent steps happen in Linear.fit."""
    P = [p for p in (probes or C.PROBES) if p in df]
    if fset in ("A", "B"):
        X = df[P].copy()
        X["open_weights"] = df.open_weights.astype(float)
        return X
    if fset == "C":
        X = pd.DataFrame({f"logit_{p}": logit(df[p]) for p in P}, index=df.index)
        for p in P:
            X[f"missing_{p}"] = df[p].isna().astype(float)
        X["t_years"] = df.t_years
        X["third_party"] = df.target_third_party.astype(float)
        return X
    raise ValueError(fset)


class Linear:
    """Ridge (alpha > 0) or OLS (alpha == 0) on logit(target), standardized features."""

    # Columns exempt from the ridge penalty when free_controls=True: the time and source terms are the
    # hypotheses under test (H1, H2), so they should not be shrunk toward zero together with the many probes.
    FREE = ("t_years", "third_party")

    def __init__(self, fset="C", alpha=1.0, k=None, drop=None, probes=None, free_controls=False):
        self.fset, self.alpha, self.k, self.drop = fset, alpha, k, drop
        self.probes, self.free_controls = probes, free_controls

    def _design(self, X):
        Z = self.scaler.transform(X)
        if self.free_controls:
            # scaling a standardized column by c divides its effective penalty by c^2; c = 1e3 makes it ~unpenalized
            for j, c in enumerate(self.cols):
                if c in self.FREE:
                    Z[:, j] *= 1e3
        return Z

    def _features(self, df):
        X = features(df, self.fset, self.probes)
        if self.drop:
            X = X[[c for c in X.columns if not c.startswith(self.drop)]]
        return X

    def fit(self, tr: pd.DataFrame):
        X = self._features(tr)
        X = X.loc[:, X.notna().any()]  # a probe never observed in the training rows carries no information
        self.mu = X.mean()
        X = X.fillna(self.mu)
        y = logit(tr.y)
        if self.fset == "B":
            r = X.apply(lambda c: np.abs(np.corrcoef(c, y)[0, 1]) if c.std() > 0 else 0.0)
            X = X[r.sort_values(ascending=False).index[: self.k]]
        self.cols = list(X.columns)
        self.scaler = StandardScaler().fit(X)
        est = Ridge(alpha=self.alpha) if self.alpha > 0 else LinearRegression()
        self.est = est.fit(self._design(X), y)
        return self

    def predict(self, te: pd.DataFrame) -> np.ndarray:
        X = self._features(te)
        if "third_party" in X:
            X["third_party"] = 0.0  # predict the vendor-reported score
        X = X.fillna(self.mu)[self.cols]
        return inv_logit(self.est.predict(self._design(X)))


class TrainMean:
    def fit(self, tr):
        self.m = tr.y.mean()
        return self

    def predict(self, te):
        return np.full(len(te), self.m)


SIMPLE_PROBE = "gpqa_diamond"


class SingleOLS:
    """logit(target) ~ logit(GPQA): the 'simple' baseline (the allowlist probe reported by most models).
    A missing GPQA is replaced by the training mean."""

    def fit(self, tr):
        x = logit(tr[[SIMPLE_PROBE]])
        self.mu = np.nanmean(x)
        ok = ~np.isnan(x).ravel()
        self.est = LinearRegression().fit(x[ok], logit(tr.y)[ok])
        return self

    def predict(self, te):
        x = logit(te[[SIMPLE_PROBE]])
        x[np.isnan(x)] = self.mu
        return inv_logit(self.est.predict(x))


class GBM:
    """Strong nonlinear baseline on Set A raw inputs (native NaN handling), plus release time."""

    def fit(self, tr):
        X = features(tr, "A").assign(t_years=tr.t_years)
        self.est = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, min_samples_leaf=3,
                                                 max_depth=3, random_state=C.SEED).fit(X, tr.y)
        return self

    def predict(self, te):
        return self.est.predict(features(te, "A").assign(t_years=te.t_years))
