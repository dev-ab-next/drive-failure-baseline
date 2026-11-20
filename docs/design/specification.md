# Design specification

What the project must do, and why. How it is built is in
[`implementation.md`](implementation.md); every parameter and the reason
for it is in [`../decisions.md`](../decisions.md). Each requirement below
names where it is met, so it can be checked.

## 1. Problem

Drive failures are rare: about 1–2% of drives fail in a year, so almost
every daily record is a healthy drive. Models trained on such data tend
to predict "healthy" all the time. Two families of fixes exist:

- **Resampling:** reshape the real data, by dropping healthy examples or
  by blending real failure examples into new ones (SMOTE).
- **Synthetic data:** train a generative model on real failures and draw
  new examples from it.

Published results on drive failure prediction are hard to compare:
studies use different data, splits, warning windows and metrics
(accuracy, F-measure, recall or detection rate, counted per day or per
drive). Splitting days at random, rather than by time, is known to leak
future information into training and make models look better than they
are (Lyu et al., *ACM TOSEM*, 2021).

## 2. Questions the project answers

1. Under one leak-free evaluation, how much do learned models improve on
   a simple SMART rule at a fixed false-alarm rate?
2. Does resampling (SMOTE) or generated synthetic failures (CTGAN) improve
   detection over a plain class-weighted baseline, when both add the same
   number of rows on identical splits?
3. How much do tuning and blending add, when every setting is chosen on
   validation and every attempt is logged?

## 3. Users

| User | Needs |
|---|---|
| Researcher | Fixed splits and metric code to compare a new method against |
| Operator | Trained baselines that reproduce the published results, and code to retrain them; scoring another fleet's data needs that data in Backblaze's format, run through the same feature pipeline |
| Reader checking the work | A README that explains the work, an evaluation method that can be read on its own, and a way to reproduce results |

## 4. Requirements and acceptance criteria

### Data

| ID | Requirement | Met by |
|---|---|---|
| D1 | Download a chosen range of quarters from Backblaze's official source, recording size and SHA-256 of every file; rerunning does not download again | `dfb prepare`, `data/manifest.json` |
| D2 | Every quarter is written in one declared schema (fixed columns, order and types); an unknown column stops conversion; every read checks the schema | `schema/drive_stats_v1.json`, `dfb validate` |
| D3 | Value rules hold: required columns non-empty, failure is 0 or 1, capacity positive where reported, dates inside the quarter, no duplicate drive-day | `schema.validate_file`, tests |
| D4 | Column changes between quarters are reported | `reports/schema_changes.md` |
| D5 | Raw Backblaze data is never committed or rehosted | `.gitignore`, README data credit |

### Scope from data analysis

| ID | Requirement | Met by |
|---|---|---|
| A1 | A failure profile per drive model, brand, capacity, age and quarter, with annualized failure rates, failure-count buckets, coverage at several cut-offs, and a check of how much each training-window length covers the test drives | `reports/failure_profile.md` |
| A2 | Scope decisions (window length, drive types, which models to train on and report) are taken from that profile and recorded | decisions D-2, R-1 to R-4 |

### Splits, labels and leakage

| ID | Requirement | Met by |
|---|---|---|
| S1 | Splits are by time: training older than validation, validation older than test. A drive may appear in several periods, because an operator always has a drive's past | decisions S-1 to S-4 |
| S2 | A day's features use only that day and earlier days | `features.py`, tests |
| S3 | A day is used only if its N-day label is decided inside its own period | decision L-2, `labels.usable` |
| S4 | Resampling and synthetic generation see training rows only | `run.py` |
| S5 | An automated check stops any run whose training or validation rows break S1 or S3, and dates, period order and post-failure rows are checked on the full data | `leakage.py`, `dfb splits`, `reports/splits.md` |
| S6 | A day is positive if the drive fails within the next N days; N = 7 is the main window, 14 and 30 are also reported | decisions L-1, R-5 |

### Evaluation

| ID | Requirement | Met by |
|---|---|---|
| E1 | Results are counted per drive: a failing drive is caught if alarmed within its window; a healthy drive is a false alarm if ever alarmed | `evaluate.py`, `EVALUATION.md` §4 |
| E2 | The main metrics are failures caught at 0.1% and 1% false alarms, with the threshold set on validation and applied unchanged to test, and days of warning | decisions B-2, E-2 |
| E3 | Also reported: PR-AUC, calibration (Brier after isotonic calibration on validation), detection by test month, spread over 3 seeds, CPU time; ROC AUC only for comparison with older work | `RESULTS.md`, `all_runs.csv` |
| E4 | Accuracy is never a result; it appears once, to show why it misleads | README, `RESULTS.md` |
| E5 | Results by brand, capacity, age band, drive model (rates only with ≥ 30 test failures) and drive models unseen in training | decisions B-6, R-3, R-4 |
| E6 | Detection across false-alarm budgets from 0.1% to 10%, labelled as using thresholds set on test | decision R-5, `RESULTS.md` |

### Models and methods

| ID | Requirement | Met by |
|---|---|---|
| M1 | A ladder from basic to advanced: always healthy, SMART rule, logistic regression, random forest, LightGBM | decision B-7 |
| M2 | Imbalance methods: class weighting, undersampling, SMOTE, CTGAN synthetic failures | decision B-8 |
| M3 | SMOTE and CTGAN run on identical splits, features and settings, adding exactly the same number of rows, and appear side by side | decision E-3, test in `tests/test_models.py` |
| M4 | Every run saves its settings, seed, code version, data window and per-drive scores, and can resume after a restart | `results/benchmark_v1/*/*/metrics.json`, `dfb bench` |
| M5 | Follow-up studies choose settings on validation only, by a rule fixed in advance, log every attempt, and keep the original results beside the new ones | decisions M-4 to M-6, `results/smote_study/`, `results/improvement_study/experiments.csv` |

### Benchmark and release

| ID | Requirement | Met by |
|---|---|---|
| B1 | A frozen benchmark definition: quarters, split dates, source checksums, schema, label rule, windows, scope and per-split counts | `benchmark/v1/benchmark.json`, `dfb freeze`, `dfb verify` |
| B2 | Anyone can score their own predictions with the same metric code | `dfb test-rows`, `dfb evaluate` |
| B3 | Every benchmark and SMOTE-study model is released with checksums, with instructions to load it and score the benchmark features (improvement-study models are rebuilt by `dfb improve`) | GitHub release v1.0, README "Use the trained models" |
| B4 | A quick check confirms the pipeline on one real quarter in about 10 minutes | `dfb quickcheck` |
| B5 | An end-to-end notebook runs from the prepared data using the same package code | `notebooks/01_end_to_end.ipynb` |

### Quality

| ID | Requirement | Met by |
|---|---|---|
| Q1 | Splitting, labelling, the leakage check, metrics and schema rules have automated tests | `tests/` |
| Q2 | Library versions are pinned and seeds fixed | `requirements.lock`, decisions |
| Q3 | Runs on a CPU-only machine with about 16 GB of memory | README "Reproduce everything" |
| Q4 | No number or rule that shapes a result is used without an entry in the decision log | `docs/decisions.md` |
| Q5 | Apache-2.0 licence; public data only; no account or agreement needed | `LICENSE`, README |

## 5. Not done in v1

- Training on synthetic data only and testing on real data, with a
  privacy distance check.
- A sequence model (LSTM or 1-D CNN) on each drive's recent history.
- A time-series generator (TimeGAN).
- A measure of how realistic the synthetic rows are.

## 6. Out of scope

- Components other than hard drives; boot drives (too few failures and
  different SMART attributes; decision R-1).
- Fleets other than Backblaze, except through the documented input
  format.
- A monitoring service or user interface.
- Claims of being the first or best predictor, or of being a community
  standard: the benchmark is a documented, reproducible evaluation
  protocol.
