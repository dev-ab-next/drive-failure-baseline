# Implementation plan

How the requirements in [`specification.md`](specification.md) are built.
Parameter values and reasons are in [`../decisions.md`](../decisions.md).

## Technical choices

| Topic | Choice | Why |
|---|---|---|
| Language | Python 3.11+ (tested on 3.13) | Standard for ML |
| Storage | Parquet, one file per quarter, zstd | 10–20× smaller than CSV, fast column reads |
| Data processing | pyarrow, pandas, DuckDB | DuckDB window functions build change features without loading a quarter into memory |
| Models | scikit-learn, LightGBM, imbalanced-learn (SMOTE), ctgan (CPU PyTorch) | Common, CPU-friendly, easy to check |
| Tests | pytest | |
| Packaging | `pyproject.toml`, exact versions in `requirements.lock` | Reproducible installs |

## Layout

```
src/dfb/
  config.py      data window, split dates, paths
  schema.py      declared schema and validation rules
  download.py    fetch quarter zips, record checksums
  convert.py     daily CSVs -> one Parquet per quarter, in the declared schema
  prepare.py     download + convert + validate every quarter
  profile.py     failure profile per drive model, brand, capacity, age, quarter
  features.py    drive table and features, by quarter and drive group
  labels.py      N-day labels and the period-boundary rule
  leakage.py     automated leakage check
  splits.py      leakage rules on the full data, per-split counts
  models.py      feature representation, model ladder, SMOTE, CTGAN
  run.py         train, calibrate, score and evaluate one run
  evaluate.py    per-drive metric code (also used by `dfb evaluate`)
  bench.py       the full run list, restart-safe
  smote_study.py SMOTE settings study (M-4) and CTGAN with weights (M-5)
  improve.py     improvement study: tuning, blending, per-brand thresholds (M-6)
  benchmark.py   freeze, verify, export test rows, score outside predictions
  report.py      RESULTS.md and all_runs.csv
  quickcheck.py  one-quarter end-to-end check
  cli.py         the `dfb` command
schema/          declared Parquet schema
benchmark/v1/    frozen benchmark definition
reports/         data profile, schema changes, split counts
results/         per-run metrics and slices, study logs, RESULTS.md
notebooks/       end-to-end notebook
tests/           unit tests on small hand-made data
```

## Data

- **Download:** each quarter's zip from Backblaze's official bucket is
  streamed to disk; size and SHA-256 go to `data/manifest.json`. A quarter
  already converted and recorded in the manifest is skipped; `dfb verify`
  compares the recorded checksums with the frozen benchmark.
- **Convert:** each zip holds one CSV per day and columns can change
  between days. Every header is checked first; a column the declared
  schema does not list stops the run before anything is written. Days are
  read one at a time, conformed to the schema (missing columns empty,
  fixed order and types, capacity −1 stored as empty) and appended to the
  quarter's Parquet file, so memory holds one day. The finished file is
  validated, then the zip is deleted.
- **Read:** every reader checks the schema in the file footer first.

## Failure profile

Reads date, serial, model and failure only. A drive's model is the one it
reports most often; a failed drive is counted once, on its first failure
day, which sets its period. Brand comes from the model-name prefix, and a
drive is a boot drive if it is an SSD or a hard drive under 2 TB.
Annualized failure rate uses every daily row, so drives that arrive or
retire mid-window are counted fairly.

## Features and labels

- **Rows:** data hard drives only.
- **Core features:** 14 SMART attributes reported by every brand; change
  over 7 and 30 days for the error counters; age; capacity; brand. The
  drive model name is not a feature, so the unseen-model slice is a fair
  test.
- **Full features:** core plus brand-specific attributes, tested as an
  add-on because they also act as brand labels.
- **Memory:** 343 million drive-days on a 15 GB machine. Features are
  built per quarter, with the previous 30 days carried over, in 8 groups
  of drives by serial-number hash, using DuckDB.
- **Training table:** every day within 30 days of a failure plus a fixed
  2% hash sample of other days, weighted to stand for the rest. Every
  method starts from this same table.
- **Validation and test:** test scores every drive-day, in slices of one
  million rows, because per-drive false alarms need every day of every
  healthy drive. Validation scores every day of a fixed 25% of drives;
  calibration and early stopping use a 2% sample of those drives' other
  days (decision E-5).

## Models and methods

| Level | Model | Settings |
|---|---|---|
| L0 | Constant score | – |
| L1 | SMART rule | Number of SMART 5, 187, 188, 197, 198 above zero |
| L2 | Logistic regression | Standardized, C = 1 |
| L3 | Random forest | 100 trees, depth ≤ 12, 20% of rows per tree, leaves ≥ 50 |
| L4 | LightGBM | Learning rate 0.1, ≤ 300 trees with early stopping on validation, 63 leaves, ≥ 100 rows per leaf, 80% feature and row sampling |

All learned models use one representation: signed log(1 + |x|), empty
values filled with the training median plus a "not reported" flag.

Methods M0–M3 are described in [`../../EVALUATION.md`](../../EVALUATION.md) §7.
Each learned model is calibrated with isotonic regression on validation.

## Runs

| Window | Runs |
|---|---|
| 7 days | L0, L1; L2 × M0–M3; L3 × M0; L4 × M0–M3; L4 × M0 with full features |
| 14 and 30 days | L1; L2 × M0; L4 × M0 |

Learned models run with 3 seeds. Every run writes its settings, metrics,
slices and per-drive scores to `results/benchmark_v1/N<window>/<run>/`.
The repository keeps every run's `metrics.json` and the slice tables of
the headline run (LightGBM, class weighting, seed 0), which `RESULTS.md`
reports. `dfb bench` skips runs already finished, so a restart loses at
most one run. Model files are published in the GitHub release.

## Studies

`dfb smote-study` and `dfb improve` score each variant on validation
only, pick by a rule fixed in advance, then run the choice on test with 3
seeds. Both skip finished work after a restart. The improvement study
appends one row per run to `results/improvement_study/experiments.csv`
with its settings, scores, timing and code version.
