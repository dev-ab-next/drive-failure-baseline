# Evaluation method (benchmark v1)

This document describes how every number in this repository is produced,
so the method can be reviewed without reading the code. Each rule names
its entry in [`docs/decisions.md`](docs/decisions.md), where the reason
for it is recorded. The frozen definition is
[`benchmark/v1/benchmark.json`](benchmark/v1/benchmark.json).

## Principles

- **No leakage.** Nothing from the test period reaches training; the
  rules below are checked automatically on every run.
- **Reproducible.** One command sequence rebuilds every number from the
  public Backblaze download; seeds are fixed and library versions pinned.
  Data, failure profile and split counts reproduce exactly; trained
  models reproduce exactly from the same feature files, and closely but
  not exactly after rebuilding them (§9).
- **Honest comparisons.** Every method uses the same splits, features,
  window and metrics; the simple baseline is always reported, and results
  that make a method look worse are kept.
- **Data terms.** Backblaze is credited; raw data is never committed or
  rehosted; nothing is sold.
- **Claims stay within the evidence.** Results are for hard drives and
  SMART data on one operator's fleet.

## 1. Task

For each hard drive and each day, give a risk score that the drive fails
within the next **N days**. N = 7 is the main window; N = 14 and N = 30 are
also reported (R-5). A day is *positive* if the drive fails within the next N
days, counting the failure day itself (L-1).

## 2. Data

- **Source:** Backblaze Drive Stats, 12 quarters, 2023 Q4 to 2026 Q3,
  downloaded from Backblaze's official server (D-1, D-2). Every source
  file's SHA-256 checksum is recorded in `benchmark/v1/benchmark.json`;
  `dfb verify` checks a local copy against it.
- **Schema:** every quarter is converted to one declared schema
  (`schema/drive_stats_v1.json`, 197 columns, fixed order and types).
  Unknown columns stop conversion; value rules are checked on every file
  (D-6, D-8).
- **Scope:** data hard drives only. Boot SSDs and boot hard drives under
  2 TB are excluded: about 160 failures in three years and none in the
  test period (R-1, F-7).
- **Counting:** a drive is a serial number; a failed drive is counted once,
  on its first flagged day; days after it are removed (F-1 to F-4).

## 3. Splits (by time)

| Split | Dates | Drives | Failed drives |
|---|---|---|---|
| Training | 2023-10-01 – 2025-09-30 | 355,607 | 8,935 |
| Validation | 2025-10-01 – 2026-03-31 | 345,686 | 1,958 |
| Test | 2026-04-01 – 2026-09-30 | 371,793 | 2,681 |

(N = 7. Exact counts for every window are in `benchmark/v1/benchmark.json`
and `reports/splits.md`.)

Validation is scored on a fixed 25% of its drives, chosen by a hash of the
serial number (about 86,000 drives; E-5). Alarm thresholds and every
study's validation scores use every day of these drives. Probability
calibration and LightGBM's early stopping use a smaller table drawn from
the same drives: every day inside a failure window, plus a fixed 2% hash
sample of the other days, weighted 50 as in training (F-12). Test is
scored on every drive and every day.

**Leakage rules** (S-4, L-2). Every run checks the training and
validation rows it uses against rules 1 and 3 and stops on a breach; test
rows are read with the same rule 3 filter. `dfb splits` checks dates, period
order and that no row follows its drive's failure, on all three periods of
the full data:

1. Periods do not overlap in time.
2. A day's features use only that day and earlier days. A unit test
   changes a drive's future values and confirms no earlier feature
   changes.
3. A day is used only if its label is decided inside its own period:
   either its whole N-day window ends inside the period, or the drive
   fails inside both the window and the period. So no label uses
   information from a later period.
4. Resampling and synthetic generation use training rows only: they run
   on the training table after the split, and validation and test rows
   are never passed to them.

A drive may appear in more than one period (for example, its 2024 days in
training and its 2026 days in test). That matches real use: an operator
always has a drive's past. Days of one drive are never mixed at random
across splits: random day-level splits are known to leak future
information into training (Lyu et al., *ACM TOSEM*, 2021).

## 4. Unit of evaluation: the drive (B-1, E-1)

Operators act on drives, not on days, so results are counted per drive
in the test period:

- A **failing drive** fails during the test period. It is **caught** if
  its highest daily score in its window (the failure day and the N − 1
  days before) is at or above the alarm threshold.
- A **healthy drive** does not fail during the test period. It is a
  **false alarm** if its highest daily score on any usable day is at or
  above the threshold.
- Alarms a failing drive raises *before* its window are ignored: they
  neither catch the failure nor count as false alarms. Published work
  differs on this point; this is our stated choice.

## 5. Metrics, in order of importance

| Metric | Definition |
|---|---|
| **Failures caught at a fixed false-alarm rate** (main) | The alarm threshold is the lowest score at which at most 0.1% (or 1%) of healthy drives raise an alarm. Detection rate = share of failing drives caught at that threshold (B-2) |
| Failures caught at other budgets | The same detection rate at budgets from 0.1% to 10%, with thresholds set on the test scores (labelled as such). Shows how detection grows as more false alarms are allowed (R-5) |
| **Days of warning** (main) | For each caught drive: days between its first alarm inside the window and its failure day. Median and quartiles reported |
| Precision-recall AUC | On per-drive scores; suited to rare positives |
| Calibration | Day-level Brier score on test after fitting isotonic calibration on validation. Undersampling and SMOTE distort raw probabilities, so raw scores are not compared (B-4) |
| ROC AUC | Reported only for comparison with older work; it flatters models when positives are rare |

**Where the threshold comes from (E-2).** The headline numbers set the
threshold on **validation** (its 25% drive sample, §3) and apply it unchanged to test, which is what
an operator could deploy. The false-alarm rate that threshold actually
gives on test is reported next to each detection rate. A second column
sets the threshold on the test results themselves, for comparison with
results reported that way.

**Accuracy is not a result here (B-10).** A model that always predicts
"healthy" is right on about 99.98% of test drive-days and catches no
failures.

**Variation.** Learned models run with 3 seeds; tables show mean ± standard
deviation (B-5). Detection is also reported by month of failure, to show
drift over the test period.

## 6. Results by slice (B-6, R-3, R-4)

Every metric is also broken down by brand, capacity, age band (SMART 9
power-on hours) and drive model, at the same global threshold. A slice
gets a detection rate only if it has **at least 30 test failures**; with
fewer, the rate is too uncertain (about ±18 percentage points at 30
failures, ±31 at 10), so the slice is listed as "too few to measure". All
failures are still counted in the overall result. A separate slice covers
**drive models never seen in training**.

## 7. Baselines and methods (B-7, B-8)

| Level | Model |
|---|---|
| L0 | Always "healthy" |
| L1 | SMART rule: number of SMART 5, 187, 188, 197, 198 above zero (E-4). 187 and 188 are reported only by Seagate |
| L2 | Logistic regression |
| L3 | Random forest |
| L4 | LightGBM |

| Method | Training rows |
|---|---|
| M0 | Class weighting only |
| M1 | Undersampling: 10 healthy rows per failure-window row |
| M2 | SMOTE: blended failure rows added until 10 healthy per failure row |
| M3 | CTGAN synthetic failure rows, trained on training failure rows only; **adds exactly as many rows as M2** (E-3) |

All learned models and methods share one feature representation (M-1).
Model settings are fixed and untuned (M-3, M-2). Training uses every day
within 30 days of a failure plus a fixed 2% sample of other days, chosen
by a hash so it is the same on every machine; sampled days carry weight 50
(F-12).

As declared, M0 trains with those weights and M1–M3 train on their rows
without them, so the benchmark's M1–M3 results also reflect that
difference. A follow-up study (M-4, M-5) tested SMOTE settings on the
validation period only, then ran the chosen setting and a CTGAN run with
the same weighting on test. Corrected = sampling weights kept with class
balancing; for SMOTE, neighbours from other drives only, with brand and
"not reported" columns copied from one parent. The original M2 and M3
results are kept and reported beside the corrected ones.

A second study (M-6) measured what tuning adds: LightGBM settings changed
one at a time and then combined, logistic regression regularisation, the
corrected SMOTE rows on the best models, a blend of the best LightGBM and
logistic regression, and per-brand thresholds. All were scored on
validation; the rule for choosing was fixed beforehand, and the three
chosen settings ran once on test. Every attempt is logged in
`results/improvement_study/experiments.csv`.

## 8. Evaluating your own method

```bash
dfb test-rows test_rows.parquet                     # the rows to score
dfb test-rows valid_rows.parquet --period valid     # optional
# produce Parquet files with columns serial_number, date, score
dfb evaluate --predictions my_test.parquet --valid-predictions my_valid.parquet
```

`dfb evaluate` uses the same metric code as this repository's own results
and refuses predictions that leave any row unscored.

## 9. Known limits

- One operator's fleet (Backblaze), hard drives only, SMART data only.
- SMART attributes differ by manufacturer; some signals exist for one
  brand only.
- The benchmark is a documented, reproducible evaluation protocol. It is
  not a community standard.
- Rebuilding the features (`dfb features`) writes their rows in a
  different order, because several threads write them, and training reads
  rows in file order. The rows are the same, but trained models depend on
  their order, so results from a fresh build differ slightly from the
  published ones.
