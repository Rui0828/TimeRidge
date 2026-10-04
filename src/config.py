"""Project-wide constants. Fixed before any modeling; changing them invalidates the frozen splits."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_BP = ROOT / "data" / "raw" / "benchpress" / "data"
PROCESSED = ROOT / "data" / "processed"
SPLITS = ROOT / "data" / "splits"
RESULTS = ROOT / "results"

# Main target (Terminal-Bench 2.0 in BenchPress) and the second target.
MAIN_TARGET = "terminal_bench"
SECOND_TARGET = "swe_bench_pro"
TARGETS = [MAIN_TARGET, SECOND_TARGET]

# Inputs: the BenchPress low-cost candidate allowlist, used as given with no selection
# (25 benchmarks; github.com/microsoft/benchpress, experiments/sec5_findings/optimal_probe/
# candidate_allowlists/user_cheap_20260505.json, the cost-aware candidate set of Zeng & Papailiopoulos 2026, Sec. 5.1).
import json as _json
ALLOWLIST_FILE = ROOT / "data" / "raw" / "benchpress_allowlist" / "user_cheap_20260505.json"
PROBES = _json.loads(ALLOWLIST_FILE.read_text())["benchmark_ids"]

# Only audited scores (BenchPress loader default).
KEEP_AUDIT = {"verified", "verified_third_party"}

# Target cells whose reference_url host is one of these are third-party measurements (uniform scaffold);
# everything else is vendor-reported. Used as a training-time control; predictions assume vendor-reported.
THIRD_PARTY_HOSTS = {"scale.com", "artificialanalysis.ai", "llm-stats.com", "www.tbench.ai", "tbench.ai"}

# Newest fraction of each target table held out as the test set.
TEST_FRAC = 0.2

# logit clipping for percentage scores, as a fraction of 1.
CLIP = 0.005

SEED = 0
