"""BenchPress (Zeng & Papailiopoulos, arXiv 2606.24020) default predictor on our own matrix.

Same method as the repo's `predict_benchpress_scores`: logit transform + column z-score + bias ALS,
rank 2, lambda 0.1. We call it on our cleaned matrix instead of the package's bundled one so that
it sees exactly the rows our temporal protocol allows.

Requires the BenchPress repo installed (`pip install -e repos/benchpress`).
"""
import contextlib
import io
import json

import numpy as np
import pandas as pd

from src import config as C

with contextlib.redirect_stdout(io.StringIO()):  # the harness prints its bundled matrix on import
    from benchpress.evaluation_harness import make_score_predictor
    from benchpress.methods.completers import complete_bias_als, complete_soft_impute

MIN_OBS_PER_BENCH = 8  # BenchPress paper-canonical column filter (>= 8 observed models)


def bench_metrics() -> dict:
    b = pd.read_csv(C.RAW_BP / "benchmarks.csv")
    out = {}
    for bid, js in zip(b.benchmark_id, b.canonical_setting_json):
        cs = json.loads(js) if isinstance(js, str) else {}
        typ = cs.get("metric_type", "pct")
        rng = cs.get("range", [0, 100] if typ == "pct" else None)
        out[bid] = {"type": typ, "range": rng, "higher_is_better": cs.get("higher_is_better", True)}
    return out


_METRICS = None


# Matrix-completion baselines from the BenchPress repo, all on the logit transform + column z-score
# (BenchPress's default pipeline). Hyperparameters are the paper's defaults.
METHODS = {
    "benchpress": (complete_bias_als, dict(rank=2, lam=0.1, normalize=False)),
    "soft_impute": (complete_soft_impute, dict(rank=2, normalize=False)),
}


def predict(train: pd.DataFrame, new_rows: pd.DataFrame, target: str, method: str = "benchpress") -> pd.Series:
    """Predict `target` for each row of `new_rows` (only the revealed cells filled in).

    train: models x benchmarks scores of models allowed for training (full rows).
    """
    global _METRICS
    if _METRICS is None:
        _METRICS = bench_metrics()
    revealed = new_rows.columns.intersection(train.columns)
    keep = train.columns[train.notna().sum() >= MIN_OBS_PER_BENCH]
    keep = keep.union([target]).union(revealed)
    X = pd.concat([train[keep], new_rows.reindex(columns=keep)])
    bids = list(X.columns)
    metric = {b: _METRICS.get(b, {"type": "pct", "range": [0, 100], "higher_is_better": True}) for b in bids}
    completer, kwargs = METHODS[method]
    fn = make_score_predictor(completer, "logit", metric=metric, benchmark_ids=bids, **kwargs)
    P = fn(X.to_numpy(dtype=float))
    j = bids.index(target)
    return pd.Series(P[len(train):, j], index=new_rows.index)


def predict_individually(train: pd.DataFrame, new_rows: pd.DataFrame, target: str,
                         method: str = "benchpress") -> pd.Series:
    """Use a fixed historical fold cutoff without sharing validation-row inputs."""
    return pd.Series(
        [predict(train, new_rows.loc[[model]], target, method=method).iloc[0]
         for model in new_rows.index], index=new_rows.index, dtype=float)
