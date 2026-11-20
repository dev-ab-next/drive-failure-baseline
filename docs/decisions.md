# Decision log: how each parameter was chosen

Every number or rule that shapes a result is recorded here with its reason
and its status. A parameter changes only by editing this file first. Status is one of **Decided**, **Provisional** (a working
choice, to be confirmed by data) or **Open**.

## Data

| ID | Parameter | Value | Why | Status |
|---|---|---|---|---|
| D-1 | Source | Backblaze Drive Stats, official download (`f001.backblazeb2.com`) | The only large public daily SMART dataset with failure labels; terms allow research use with credit | Decided |
| D-2 | Window | 12 quarters, 2023 Q4 – 2026 Q3 | The most recent 3 years. 2026 Q3 was published by 2026-11-18, so the window ends there. The failure profile (§9) showed that 0.5 to 2 years of training all cover 98% of test drives' models, while 2 years gives about 4× the failure examples of 0.5 years (9,093 vs 2,307). Rare failures make examples the limiting factor, so the window is not shortened | Decided (confirmed by profile) |
| D-3 | Raw data handling | Downloaded, converted to Parquet, zip deleted; never committed | Backblaze terms (no rehosting); saves disk | Decided |
| D-4 | Column types | `smart_*` as float64; `capacity_bytes` int64; `failure` int8; others string | Raw SMART values can exceed 32-bit integers, and missing values must stay missing | Decided |
| D-5 | Schema changes | Missing columns are present and empty; changes listed in `reports/schema_changes.md` | Backblaze adds SMART columns over time: 2024 Q2 added SMART 211 and 212 | Decided |
| D-6 | Declared Parquet schema | `schema/drive_stats_v1.json`: 197 columns with fixed order and types, tagged in each file. Unknown columns stop conversion; every read checks the schema first; validation rules: required columns non-empty, `failure` ∈ {0, 1}, capacity > 0 where reported (D-8), dates inside the file's quarter, no duplicate (date, serial) | A silent change in a new Backblaze release (a renamed or retyped column) would otherwise change results without warning. A new column means a new schema version, recorded here | Decided |
| D-7 | Quarters without SMART 211/212 | Keep 2023 Q4 and 2024 Q1 | Those two quarters lack only SMART 211 and 212. In 2026 Q3 those columns have a value in 0.003% of rows, all from two Micron boot SSD models, which are excluded anyway (R-1). Dropping the quarters would remove about 2 of the 8 training quarters and their failures and gain nothing. Rule for later: a SMART column with almost no values is dropped as a feature, not the quarters that lack it | Decided |
| D-8 | Capacity of -1 | Stored as empty ("not reported that day"); the count per quarter goes in the manifest. Any other value ≤ 0 fails validation | Found by the schema validation: Backblaze writes -1 on some days (8,390 rows in 2023 Q4, on 8,307 drives, none on a failure day), and every such drive reports its real capacity on other days. A drive's capacity is taken from the days it is reported | Decided |

## Definitions

| ID | Parameter | Value | Why | Status |
|---|---|---|---|---|
| F-1 | A drive | One serial number | Backblaze's own unit | Decided |
| F-2 | A drive's model | The model label it reports on the most days | A few serials change label slightly over time; the most common label is the stable one | Decided |
| F-3 | A failed drive | Any day with `failure == 1`; counted once | A few drives are flagged on more than one day; counting days would double-count | Decided |
| F-4 | Failure date | First flagged day | The earliest sign is what a predictor must beat | Decided |
| F-5 | Failure rate | AFR = failures ÷ (drive-days ÷ 365) × 100% | Backblaze's published formula, so our numbers are comparable to its reports. Raw failure counts mislead because big models have more failures simply by having more drives | Decided |
| F-6 | Brand | From the model-name prefix (e.g. `ST` → Seagate, `HGST`/`Hitachi` → HGST, `WDC`/`WU` → Western Digital) | Backblaze has no brand column. Unmatched names are listed in the report so the rule can be fixed | Decided |
| F-7 | Drive type | **Boot SSD** if the name marks an SSD (contains `SSD`, `SA510`, `WDS…`, `ZA…`, or starts with `MTFD`, `Micron`, `CT`, `DELLBOSS`, `SSDSC`, `MZ`/`Samsung MZ`); else **boot HDD** if under 2 TB; else **data HDD** | Backblaze runs small boot drives (SSDs and 2.5-inch 250–500 GB hard drives) next to its data drives. They have different SMART attributes and failure behaviour, so mixing them would distort results. Revised after the first full profile found small SSDs and 500 GB boot hard drives counted as data drives | Decided (revised 2026-11-18) |
| F-8 | Drive age | SMART 9 (power-on hours) ÷ 8,760 | The only age signal in the data. Bands: 0–1, 1–2, 2–3, 3–5, 5–7, 7+ years | Decided |

## Features and labels

| ID | Parameter | Value | Why | Status |
|---|---|---|---|---|
| F-9 | Core SMART columns | Raw values of 1, 3, 4, 5, 7, 9, 10, 12, 192, 193, 194, 197, 198, 199 | Reported by every brand on 93–100% of data-HDD rows (3 sample days of 2026 Q2, 1,040,419 rows) | Decided |
| F-10 | Change features | Change over the last 7 and 30 days for SMART 5, 187, 188, 197, 198, 199: today's value minus the earliest value in that window | A rising error count is a stronger warning than a fixed one; Backblaze notes a jump from 0 to 20 in a day is more concerning than a steady value. "Earliest in the window" tolerates missing days | Decided |
| F-11 | Brand-specific columns | 2, 8, 18, 22, 23, 24, 27, 71, 90, 184, 187, 188, 189, 190, 191, 195, 196, 200, 220, 222, 223, 224, 226, 240, 241, 242; only in the F-full set | Reported by one or a few brands only (e.g. 187 and 188: Seagate only), so they also act as brand labels; tested as an add-on | Decided |
| F-12 | Training sample | Every day within 30 days of a failure, plus 2% of other days chosen by a fixed hash of (serial, date, "dfb-benchmark-v1"); sampled days carry weight 50 | 216 million training drive-days do not fit in 15 GB. A fixed hash gives the same sample on every machine. The weight restores the true failure rate for calibration. All methods start from this same table | Decided |
| F-13 | Model name as a feature | Not used; brand is used | Keeps the new-drive-model slice (R-4) a fair test | Decided |
| L-1 | Label | Positive if the drive fails within the next N days, counting the failure day (days to failure 0 to N−1). Days after the failure are removed | Matches "fails within N days"; N = 7 main, 14 and 30 also reported (R-5) | Decided |
| L-2 | Which rows are used | A row is used only if its label is decided inside its own period: its N-day window ends inside the period, or the drive fails inside both the window and the period. Other rows near a period end (outcome unknown, or decided by a later period) are dropped | First version dropped every row in the last N−1 days of a period; on the real data that lost 65 test failures (2,616 instead of 2,681) whose labels needed nothing from later. The revised rule keeps them and is still leak-free | Decided (revised 2026-11-18) |

## Splits

| ID | Parameter | Value | Why | Status |
|---|---|---|---|---|
| S-1 | Split method | By time | Random day-level splits mix days of the same drive across training and test, which leaks future information into training (Lyu et al., ACM TOSEM 2021) | Decided |
| S-2 | Proportions | Train 8 quarters, validation 2, test 2 | The test set must hold enough failures to measure detection rates; 2 quarters is the shortest period that does. Validation the same size, for tuning | Decided |
| S-3 | Dates | Train 2023-10-01 – 2025-09-30; validation 2025-10-01 – 2026-03-31; test 2026-04-01 – 2026-09-30 | Follows from D-2 and S-2; boundaries fall on quarter starts | Decided |
| S-4 | Drives across periods | Allowed. Guards: a row is used only if its label is decided inside its own period (L-2); features look backward only; resampling and synthesis inside training only | Most drives run all 12 quarters. Keeping drives disjoint as well would cut test failures from 2,681 to about 400 and leave about 5 models with 30+ test failures. An operator always has a drive's past, so allowing it matches real use | Decided |

## Analysis thresholds (failure profile)

| ID | Parameter | Value | Why | Status |
|---|---|---|---|---|
| A-1 | Failure buckets | 0, 1–9, 10–29, 30–99, 100–299, 300+ | Roughly a factor of 3 per step, so a few buckets span the full range. 30 is the usual rule-of-thumb minimum for a rate to be stable | Decided |
| A-2 | Coverage cut-offs | 10, 30, 100, 300 failures | Bucket edges, so the coverage table reads alongside the buckets | Decided |
| A-3 | Window-check threshold | A model counts as covered if it has ≥ 10 training failures | Below about 10 examples a model learns little about that drive model's failure pattern | Provisional |
| A-4 | Window-check windows | 0.5, 1, 1.5, 2 years of training, all ending 2025 Q3 | Training is 2 years at most within 12 quarters (S-2); halving steps show where coverage starts to drop | Decided |

## Benchmark protocol

| ID | Parameter | Value | Why | Status |
|---|---|---|---|---|
| B-1 | Unit of evaluation | Per drive: a failing drive is caught if alarmed within the warning window before failure; a healthy drive is a false alarm if alarmed at any time in the test period | How operators experience a predictor. Papers count this differently, so the rule is stated explicitly | Decided |
| B-2 | False-alarm rates | 0.1% and 1% of healthy drives | Papers report detection at very low false-alarm rates; 1% shows the trade-off at a looser budget | Decided |
| B-3 | Main metrics | Detection rate at B-2, and precision-recall AUC | Suited to rare failures; ROC AUC kept only for comparison with older papers, since it flatters rare-event models | Decided |
| B-4 | Calibration | Brier score after recalibration on the validation set | Undersampling and SMOTE distort predicted probabilities, so raw scores would not be comparable | Decided |
| B-5 | Seeds | 3 per run; mean and spread reported | Separates real differences from luck at modest compute cost | Decided |
| B-6 | Slices | Brand, capacity, age band, drive model, new drive model | Shows where the model works and where it does not. Drive type is fixed to data hard drives by R-1 | Decided; per-model rule R-3 |
| B-7 | Model ladder | L0 trivial, L1 SMART rule, L2 logistic regression, L3 random forest, L4 LightGBM. A sequence neural network (L5) was planned but not built in v1 (`docs/design/specification.md` §5) | Shows how much each step up in model complexity adds. L1 uses SMART 5, 187, 188, 197, 198 (Backblaze's named failure signals) | Decided |
| B-8 | Method crossing | M1–M3 with L2 and L4 only | Keeps the run count within the v1 compute budget (about a day on 4 CPUs). **Revised 2026-11-20:** as first written, M3 was to run with the better of L2 and L4; it was run with L4 only, although L2 was better on validation and test. L2 × M3 (3 seeds) was then added so both models are covered; it is reported whatever it shows. Result (first setup, no sampling weights): 42.7 ± 15.8% at 1% false alarms, seeds 53.4%, 24.5% and 50.1%, against 55.0% for L2 × M0. CTGAN with weights (M-5) was run with L4 only | Decided (revised) |
| B-9 | Deferred | TimeGAN (M4) | Left for future work: heavy on CPU | Decided |
| B-10 | Order of results | Failures caught at fixed false-alarm rates and days of warning first; accuracy never reported as a result, shown once to explain why it misleads | Accuracy is near 100% for a model that predicts no failures at all, so it says nothing about rare events | Decided |
| B-11 | Benchmark freezing | `benchmark/v1/`: split dates, source-zip checksums, schema, label rule, scope, per-split counts; `dfb evaluate` scores outside predictions with the same code | So other methods can be measured on exactly the same data and metrics | Decided |

## Evaluation and baselines

| ID | Parameter | Value | Why | Status |
|---|---|---|---|---|
| E-1 | Early alarms on failing drives | Ignored: a failing drive is caught if its highest score in its N-day window crosses the threshold; its alarms before the window are not false alarms. A healthy drive's score is its highest over the period | Simple and common; an early alarm on a drive that does fail is not a false alarm in practice. Stated in `EVALUATION.md` because papers differ | Decided |
| E-2 | Alarm threshold | Both reported; **headline uses the threshold set on validation** and applied unchanged to test, with the false-alarm rate it actually gives on test. Threshold set on the test curve shown alongside, for comparison with papers | The validation threshold is what an operator could really deploy | Decided |
| E-3 | SMOTE vs synthetic | M2 and M3 both top up to 10 healthy rows per failure-window row, adding exactly the same number of rows | So M2 and M3 differ only in how the extra rows are made | Decided |
| E-4 | SMART rule (L1) score | Number of SMART 5, 187, 188, 197, 198 above zero (0 to 5). 187 and 188 exist only on Seagate | Mirrors Backblaze's "non-zero is a reason to investigate"; a count gives a few operating points instead of one | Decided |
| E-5 | Validation scoring | A fixed 25% of drives (by serial hash). Thresholds and study scores use every day of each; calibration and LightGBM early stopping use every failure-window day of those drives plus a fixed 2% hash sample of their other days, weighted 50 (as F-12). Test uses every drive and day | Saves scoring and fitting time; thresholds and calibration still rest on ~86,000 drives | Decided |

## Models and methods

| ID | Parameter | Value | Why | Status |
|---|---|---|---|---|
| M-1 | One feature representation for all learned models and methods | F-core columns; signed log(1 + \|x\|); empty → training median plus a "not reported" flag | SMOTE and CTGAN cannot handle empty values. Using the same representation for M0–M3 means the methods differ only in the rows they train on | Decided |
| M-2 | CTGAN settings | ctgan 0.11.0 defaults (generator and discriminator 256×256, learning rate 2e-4), 100 epochs, batch 500, CPU; trained on training failure-window rows only; flags and brand columns treated as categories | Default architecture avoids tuning the generator to the test set; 100 epochs keeps each run under about 30 minutes on CPU. Recorded per run | Decided |
| M-3 | Model settings | L2: logistic regression C = 1, standardized, at most 300 solver iterations. L3: random forest, 100 trees, depth ≤ 12, 20% rows per tree, leaves ≥ 50. L4: LightGBM, learning rate 0.1, at most 300 trees, 63 leaves, ≥ 100 rows per leaf, 80% feature and row sampling, L2 regularisation 1, early stopping (30 rounds, log loss) on the validation sample (E-5) | Common, untuned settings. **Revised 2026-11-18 before any test result was seen:** the first LightGBM run (learning rate 0.05, up to 2,000 trees) used all 2,000 trees, and scoring the 77 million validation and test rows would take about 2 hours per run. The run was stopped during scoring, its model deleted, and the caps above set; random-forest depth capped for the same reason | Decided (revised) |
| M-4 | SMOTE settings study | After the benchmark showed SMOTE lowering LightGBM detection (43.3% vs 52.1%), 13 variants are compared: M0 and the original M2 as references; SMOTE with k = 3, 5, 10, 25 neighbours; top-up ratios 10:1, 25:1, 50:1, 100:1; neighbours from other drives only; brand and "not reported" columns copied instead of blended; F-12 sampling weights kept with class balancing; and the last three combined. LightGBM, 7 days, core features, seed 0. **Chosen on validation only** (failures caught at 1% false alarms, threshold set on validation); the chosen variant then runs on test with 3 seeds | Tests four reasons SMOTE could hurt: near-copies of one drive (most neighbours are the same drive's other days), impossible blended rows, too many synthetic rows, and the missing sampling weights. Selecting on validation keeps test untouched. The pre-declared M2 result stays the benchmark's SMOTE result; the study is reported in full beside it, whatever it shows | Decided |
| M-5 | CTGAN with sampling weights | M3 rerun on test (3 seeds) with the F-12 sampling weights kept and classes balanced, as in the best M-4 variant. Decided after the M-4 validation results and before any M-5 result | M-4 found that training without the sampling weights was the largest single cause of SMOTE's loss, and M3 was trained the same way. Comparing a fixed SMOTE with an unfixed CTGAN would not be fair | Decided |
| M-6 | Improvement study | 7 days, core features. **A** LightGBM: benchmark settings (A1), then one change at a time (31 or 127 leaves; at least 500 or 20 rows per leaf; learning rate 0.05 with up to 600 trees; L2 regularisation 10; 50% of features per tree), then every improving change combined (A9). **B** logistic regression with C = 1, 0.1, 10. **C** the best of A and of B, each with the corrected SMOTE rows (M-4). **D** a blend (mean calibrated probability) of the best LightGBM and best logistic regression; then per-brand thresholds on the best so far. Stages A–D are scored on validation only (seed 0; threshold set on validation). **T** the best LightGBM, the best logistic regression and the best overall, by validation detection at 1% false alarms (PR-AUC breaks ties), run on test with 3 seeds, thresholds from validation. Every attempt is logged, never overwritten, in `results/improvement_study/experiments.csv` | Benchmark settings were fixed and untuned (M-3); this measures what tuning, blending and per-brand thresholds add, without touching test until the choice is made. Not tried here: more healthy training days, new history features, drive-model information, sequence models (each needs features rebuilt or changes the benchmark's unseen-model test) | Decided |

## Scope and reporting (decided from the failure profile, 2026-11-18)

| ID | Parameter | Value | Why | Status |
|---|---|---|---|---|
| R-1 | Drive scope | Data hard drives only; boot SSDs and boot HDDs excluded from modeling | About 160 boot-drive failures in 3 years, and none in the test period; different SMART attributes | Decided |
| R-2 | Training models | All 67 data hard-drive models, no minimum failure count | 99.99% of data drives needs 58 of 67 models anyway; small models add few rows and cost nothing; dropping them helps nothing | Decided |
| R-3 | Per-model reporting | Detection rates only for models with ≥ 30 test failures (22 models, 97.2% of test failures). Others listed with counts as "too few to measure" and pooled into an "other models" row. Overall, brand, capacity and age results count every test failure | 99.99% of 2,681 test failures leaves out none, which would mean reporting models with 1 failure, whose rate can only be 0% or 100%. Approximate 95% margin on a detection rate: ±31 points at 10 failures, ±18 at 30, ±10 at 100 | Decided |
| R-4 | New-model slice | Test drives whose model never appears in training are reported separately | Tests how the model handles an unseen drive model, e.g. `WDC WUH722626ALE6L4` (26 TB, 7,219 drives, first seen 2025 Q4) | Decided |
| R-5 | Warning window and false-alarm budgets | 7 days is the main window; 14 and 30 days are also reported for L1, L2 × M0 and L4 × M0 (3 seeds). A table of failures caught at false-alarm budgets from 0.1% to 10% is added, with thresholds set on the test scores and labelled as such | 7 days is the common window in published work and enough time to copy data and replace a drive. Longer windows show how much earlier warning costs. The budget table shows the trade-off an operator chooses; the headline budgets (0.1%, 1%, set on validation) are unchanged. 14 days was added after the 7- and 30-day results were seen; it is reported in full, whatever it shows | Decided |

## Still open

None.
