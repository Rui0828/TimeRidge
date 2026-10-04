# Time-Blind Predictors Underpredict New Models

NYCU Data Mining 2026, HW1 (linear regression).

Given a newly released LLM's scores on cheap benchmarks, predict its score on an expensive agentic benchmark,
using only models released earlier. We study 20 agentic benchmarks and use Terminal-Bench 2.0 as the
representative case. Inputs are the 25 benchmarks of BenchPress's low-cost candidate list, used as given.
Predictors that ignore release time underpredict new models; TimeRidge, a ridge regression with an
unpenalized release-time term, removes the underprediction but overcorrects on the newest models.

## Layout

```
data/raw/        downloaded data, never edited (BenchPress score matrix and low-cost list)
data/splits/     frozen release-order test sets (20 benchmarks)
data/processed/  cleaned tables, rebuilt by the scripts
src/             config, data cleaning, models, BenchPress baseline wrapper
scripts/         numbered pipeline steps
results/         outputs (predictions, metrics, coefficients, figures), rebuilt by the scripts
```

## Reproduce

```bash
pip install -r requirements.txt
git clone https://github.com/microsoft/benchpress && pip install -e benchpress   # BenchPress and Soft-Impute baselines

python scripts/01_build_tables.py           # cleaned tables for Terminal-Bench 2.0 and SWE-Bench Pro
python scripts/02_freeze_splits.py          # test sets (already frozen in data/splits; never overwritten)
python scripts/03_efa.py                    # exploratory analysis and Figure 1, development split only
python scripts/04_source_check.py           # time effect under score-source controls (H1, H2)
python scripts/05_cv.py terminal_bench      # development cross-validation, every method
python scripts/05_cv.py swe_bench_pro
python scripts/06_test.py                   # test evaluation on the two benchmarks above
python scripts/07_generalization_freeze.py  # tables for the other 18 benchmarks (splits already frozen)
python scripts/08_generalization.py         # TimeRidge, TimeRidge - time, OLS on GPQA, BenchPress on the 18
python scripts/09_generalization_full.py    # remaining baselines and ablations on the 18
python scripts/10_pooled_test.py            # pool test predictions over all 20 benchmarks (Tables 3-4)
python scripts/11_pooled_cv.py              # pool development CV over all 20 benchmarks (Tables 3-4)
python scripts/12_diagnostics.py            # why the correction overshoots (Section 8)
python scripts/13_coefficients.py           # coefficients, bootstrap CIs, VIF (Figure 4)
python scripts/14_figures.py                # Figures 2-4
```

Scripts that evaluate test sets refuse to overwrite existing results.

## Where the numbers in the paper come from

| Paper | File |
|---|---|
| Tables 3 and 4, test | `results/pooled_test/pooled.csv`, `per_target.csv` |
| Tables 3 and 4, CV | `results/pooled_cv/pooled.csv`, `per_target.csv` |
| Section 8, Terminal-Bench CV and test; Table 5 | `results/cv/terminal_bench/summary.csv`, `results/test/terminal_bench/` |
| Figure 3 | `results/pooled_test/per_target.csv` |
| Figure 4 and Section 8 coefficients | `results/coefficients/terminal_bench.csv`, `terminal_bench_vif.csv` |
| Section 8 overshoot analysis | `results/diagnostics/` |
| Section 4 (EFA) | `results/efa/` |

## Data and licenses

- BenchPress score matrix (Zeng & Papailiopoulos, 2026), https://huggingface.co/datasets/microsoft/benchpress-score-matrix,
  CDLA-Permissive-2.0 (`data/raw/benchpress/data/LICENSE-CDLA-2.0.md`). Downloaded 2026-09-30.
- BenchPress low-cost candidate list, `data/raw/benchpress_allowlist/user_cheap_20260505.json`,
  from https://github.com/microsoft/benchpress (MIT).
