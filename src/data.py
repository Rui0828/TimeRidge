"""Load and clean the BenchPress score matrix.

Cleaning rules (decided before modeling; see notes/data_rules.md for rationale):
  R1 keep only scores with audit_status in KEEP_AUDIT
  R2 release_date: YYYY-MM-DD kept; YYYY-MM mapped to the 15th and flagged; year-only or missing dropped
  R3 cells whose reported setting does not match the model's canonical setting are kept and flagged
     (matches_canonical == False); §8 re-runs the analysis without them
  R4 a model enters a target table only if the target and at least one probe are observed
  R6 target cells from third-party hosts (THIRD_PARTY_HOSTS) are flagged; kept, used as a control
"""
import numpy as np
import pandas as pd

from src import config as C


def load_scores() -> pd.DataFrame:
    s = pd.read_csv(C.RAW_BP / "scores_all.csv")
    return s[s.audit_status.isin(C.KEEP_AUDIT)].copy()  # R1


def load_models() -> pd.DataFrame:
    m = pd.read_csv(C.RAW_BP / "models.csv")
    d = m.release_date.astype(str)
    full = d.str.fullmatch(r"\d{4}-\d{2}-\d{2}")
    month = d.str.fullmatch(r"\d{4}-\d{2}")
    m["date_imputed"] = month
    m["release_date"] = np.where(full, d, np.where(month, d + "-15", None))  # R2
    m = m.dropna(subset=["release_date"])
    m["release_date"] = pd.to_datetime(m.release_date)
    m["t_years"] = (m.release_date - pd.Timestamp("2025-01-01")).dt.days / 365.25
    return m.set_index("model_id")


def score_matrix(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide models x benchmarks matrix of scores, and a same-shaped non-canonical flag matrix."""
    W = scores.pivot(index="model_id", columns="benchmark_id", values="score")
    nc = scores.assign(nc=scores.matches_canonical.eq(False)).pivot(
        index="model_id", columns="benchmark_id", values="nc")
    return W, nc.reindex_like(W).astype("boolean").fillna(False).astype(bool)


def target_table(target: str, W: pd.DataFrame, nc: pd.DataFrame, models: pd.DataFrame,
                 scores: pd.DataFrame) -> pd.DataFrame:
    """One row per model with the target and at least one probe observed (R4), sorted by release date."""
    cols = [target] + [p for p in C.PROBES if p in W.columns]
    t = W[cols].join(models[["model_name", "provider", "release_date", "t_years", "date_imputed",
                              "is_reasoning", "open_weights"]], how="inner")
    t = t.dropna(subset=[target])
    t = t[t[[p for p in C.PROBES if p in t]].notna().any(axis=1)]
    t["target_noncanonical"] = nc.loc[t.index, target]
    t["probe_noncanonical_any"] = nc.loc[t.index, [c for c in cols if c != target]].any(axis=1)
    host = scores[scores.benchmark_id == target].set_index("model_id").reference_url.str.extract(
        r"https?://([^/]+)")[0]
    t["target_source_host"] = host.reindex(t.index)
    t["target_third_party"] = t.target_source_host.isin(C.THIRD_PARTY_HOSTS)  # R6
    t = t.rename(columns={target: "y"})
    return t.sort_values(["release_date", "model_name"])


def logit(pct, clip=C.CLIP):
    p = np.clip(np.asarray(pct, dtype=float) / 100, clip, 1 - clip)
    return np.log(p / (1 - p))


def inv_logit(z):
    return 100 / (1 + np.exp(-np.asarray(z, dtype=float)))
