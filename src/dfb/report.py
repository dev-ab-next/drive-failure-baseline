"""Results tables for the README (B-10): rare-event metrics first,
mean and spread over seeds, nothing about accuracy except one line."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .evaluate import threshold_for
from .run import RESULTS

HEADLINE_RUN = "L4-M0-core"  # chosen before any results: slices are shown for it
LEVEL_NAMES = {"L0": "L0 always healthy", "L1": "L1 SMART rule", "L2": "L2 logistic regression",
               "L3": "L3 random forest", "L4": "L4 LightGBM"}
METHOD_NAMES = {"M0": "class weighting", "M1": "undersampling", "M2": "SMOTE",
                "M3": "CTGAN synthetic failures"}


def load() -> pd.DataFrame:
    rows = []
    for f in sorted(RESULTS.glob("N*/*/metrics.json")):
        d = json.loads(f.read_text())
        r, m = d["run"], d["metrics"]
        row = {"n": r["n"], "level": r["level"], "method": r["method"],
               "features": r["feature_set"], "seed": r["seed"], "run_dir": f.parent.name,
               "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"],
               "brier": m.get("brier_test_calibrated"),
               "accuracy_always_healthy": m.get("accuracy_of_always_healthy_test_days"),
               "fit_seconds": r.get("fit_seconds"), "seconds": r.get("seconds")}
        for key in ("0.1pct", "1pct"):
            v, t = m[f"valid_threshold_{key}"], m[f"test_curve_{key}"]
            row[f"det_{key}"] = v["detection_rate"]
            row[f"far_{key}"] = v["false_alarm_rate"]
            row[f"warn_{key}"] = v["warning_days_median"]
            row[f"det_testcurve_{key}"] = t["detection_rate"]
        rows.append(row)
    return pd.DataFrame(rows)


def pm(s: pd.Series, pct: bool = True, digits: int = 1) -> str:
    s = s.dropna()
    if s.empty:
        return "–"
    f = 100 if pct else 1
    if len(s) == 1:
        return f"{s.iloc[0] * f:.{digits}f}"
    return f"{s.mean() * f:.{digits}f} ± {s.std(ddof=1) * f:.{digits}f}"


def table(df: pd.DataFrame, n: int) -> str:
    d = df[df["n"] == n]
    lines = ["| Model | Method | Features | Seeds | Caught at 0.1% false alarms | "
             "(actual false-alarm %) | Caught at 1% false alarms | (actual false-alarm %) | "
             "Median days of warning (1%) | PR-AUC | Caught at 1%, test-curve threshold | "
             "ROC AUC (legacy) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    order = {k: i for i, k in enumerate(["L0", "L1", "L2", "L3", "L4"])}
    groups = sorted(d.groupby(["level", "method", "features"]),
                    key=lambda kv: (order[kv[0][0]], kv[0][1], kv[0][2]))
    for (lv, me, fs), g in groups:
        lines.append(
            f"| {LEVEL_NAMES[lv]} | {METHOD_NAMES[me] if lv not in ('L0', 'L1') else '–'} | "
            f"{fs if lv not in ('L0', 'L1') else '–'} | {len(g)} | "
            f"{pm(g['det_0.1pct'])}% | {pm(g['far_0.1pct'], digits=2)} | "
            f"{pm(g['det_1pct'])}% | {pm(g['far_1pct'], digits=2)} | "
            f"{pm(g['warn_1pct'], pct=False)} | {pm(g['pr_auc'], pct=False, digits=3)} | "
            f"{pm(g['det_testcurve_1pct'])}% | {pm(g['roc_auc'], pct=False, digits=3)} |")
    return "\n".join(lines)


SWEEP_FARS = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10)
SWEEP_RUNS = (("L1", "M0"), ("L2", "M0"), ("L4", "M0"))


def budget_sweep() -> str:
    """Failures caught over a range of false-alarm budgets (R-5). Thresholds are set on
    the test scores themselves, so this shows the shape of the trade-off; the headline
    tables use validation thresholds."""
    lines = ["| Window | Model | " + " | ".join(f"{f * 100:g}%" for f in SWEEP_FARS) + " |",
             "|---|---|" + "---|" * len(SWEEP_FARS)]
    for n in (7, 14, 30):
        for lv, me in SWEEP_RUNS:
            per_seed = []
            for f in sorted((RESULTS / f"N{n}").glob(f"{lv}-{me}-core-seed*/test_drive_scores.parquet")):
                d = pd.read_parquet(f, columns=["failing", "score"])
                fail = d.loc[d["failing"], "score"].to_numpy()
                healthy = d.loc[~d["failing"], "score"].to_numpy()
                per_seed.append([(fail >= threshold_for(healthy, far)).mean() * 100
                                 for far in SWEEP_FARS])
            if per_seed:
                cells = np.mean(per_seed, axis=0)
                lines.append(f"| {n} days | {LEVEL_NAMES[lv]} | "
                             + " | ".join(f"{c:.0f}%" for c in cells) + " |")
    return "\n".join(lines)


def smote_study_section() -> list[str]:
    """Decisions M-4 and M-5: the SMOTE settings study and the corrected methods."""
    root = RESULTS.parent / "smote_study"
    if not (root / "validation.csv").exists():
        return []
    v = pd.read_csv(root / "validation.csv")
    lines = ["## Why the first SMOTE setup lowered detection (settings study)", "",
             "Decision M-4. LightGBM, 7 days, core features, seed 0. Each variant is scored on "
             "the **validation** period only (threshold set on validation); the best SMOTE "
             "variant was then run once on test. \"Same-drive pairs\" is the share of new rows "
             "made by blending a failing drive with one of its own other days. The training "
             "table already holds about 75 healthy rows per failure row, so at 100:1 nothing is "
             "added: that row is LightGBM with neither sampling weights nor added rows.", "",
             "| Variant | Rows added | Same-drive pairs | Blended brand or flag | "
             "Caught at 1% (validation) | Caught at 0.1% (validation) | PR-AUC (validation) |",
             "|---|---|---|---|---|---|---|"]
    def pct(x):
        return "–" if pd.isna(x) else f"{x * 100:.1f}%"
    for _, r in v.iterrows():
        added = "–" if pd.isna(r.get("rows_added")) else f"{int(r['rows_added']):,}"
        lines.append(f"| {r['variant']} | {added} | {pct(r.get('same_drive_parent_share'))} | "
                     f"{pct(r.get('rows_with_blended_flag_or_brand'))} | "
                     f"{pct(r['valid_caught_1pct'])} | {pct(r['valid_caught_0.1pct'])} | "
                     f"{r['valid_pr_auc']:.3f} |")
    test = []
    for f in sorted((root / "test").glob("N7/*/metrics.json")):
        d = json.loads(f.read_text())
        m = d["metrics"]
        test.append({"method": d["run"]["method"],
                     "det_1pct": m["valid_threshold_1pct"]["detection_rate"],
                     "far_1pct": m["valid_threshold_1pct"]["false_alarm_rate"],
                     "det_0.1pct": m["valid_threshold_0.1pct"]["detection_rate"],
                     "warn_1pct": m["valid_threshold_1pct"]["warning_days_median"],
                     "pr_auc": m["pr_auc"]})
    if test:
        t = pd.DataFrame(test)
        base = load()
        base = base[(base["n"] == 7) & (base["level"] == "L4") & (base["features"] == "core")]
        rows = [("Class weighting (M0)", base[base["method"] == "M0"]),
                ("SMOTE, first setup (M2)", base[base["method"] == "M2"]),
                ("CTGAN, first setup (M3)", base[base["method"] == "M3"]),
                ("SMOTE, other-drive neighbours + weights (M-4)", t[t["method"] == "M2"]),
                ("CTGAN + weights (M-5)", t[t["method"] == "M3"])]
        lines += ["", "**Test period**, LightGBM, 7 days, threshold set on validation, 3 seeds. "
                  "Corrected = sampling weights kept with class balancing (both), and for SMOTE "
                  "neighbours from other drives with brand and flag columns copied. Both add the "
                  "same number of rows.", "",
                  "| Method | Caught at 1% false alarms | (actual false-alarm %) | "
                  "Caught at 0.1% false alarms | Median days of warning | PR-AUC |",
                  "|---|---|---|---|---|---|"]
        for name, g in rows:
            lines.append(f"| {name} | {pm(g['det_1pct'])}% | {pm(g['far_1pct'], digits=2)} | "
                         f"{pm(g['det_0.1pct'])}% | {pm(g['warn_1pct'], pct=False)} | "
                         f"{pm(g['pr_auc'], pct=False, digits=3)} |")
    return lines + [""]


def improvement_section() -> list[str]:
    """Decision M-6: every tuning attempt, in the order run, then the chosen configs on test."""
    root = RESULTS.parent / "improvement_study"
    f = root / "experiments.csv"
    if not f.exists():
        return []
    e = pd.read_csv(f)
    v = e[e["period"] == "valid"]
    lines = ["## Improvement study (decision M-6)", "",
             "Every attempt, in the order it was run, copied from "
             "`results/improvement_study/experiments.csv` (one row per run, never overwritten). "
             "7 days, core features. Stages A–D are scored on the **validation** period only "
             "(seed 0, threshold set on validation). The validation threshold sample (a fixed "
             "25% of drives) holds 454 failing drives, so one drive is 0.22 points and "
             "differences under about 3 points are within chance.", "",
             "| ID | Stage | What changed | Thresholds | Caught at 1% (validation) | "
             "Caught at 0.1% (validation) | PR-AUC (validation) | Fit seconds |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in v.iterrows():
        lines.append(f"| {r['id']} | {r['stage']} | {r['description']} | {r['thresholds']} | "
                     f"{r['caught_1pct'] * 100:.1f}% | {r['caught_0.1pct'] * 100:.1f}% | "
                     f"{r['pr_auc']:.3f} | {r['fit_seconds']:.0f} |")
    t = e[e["period"] == "test"]
    if len(t):
        base = load()
        base = base[(base["n"] == 7) & (base["features"] == "core") & (base["method"] == "M0")]
        rows = [("LightGBM, benchmark settings (L4-M0)", base[base["level"] == "L4"]
                 .rename(columns={"det_1pct": "c1", "far_1pct": "f1", "det_0.1pct": "c01",
                                  "warn_1pct": "w"})),
                ("Logistic regression, benchmark settings (L2-M0)", base[base["level"] == "L2"]
                 .rename(columns={"det_1pct": "c1", "far_1pct": "f1", "det_0.1pct": "c01",
                                  "warn_1pct": "w"}))]
        for i, g in t.groupby("id", sort=False):
            rows.append((f"{i}: {g['description'].iloc[0]}",
                         g.rename(columns={"caught_1pct": "c1", "far_1pct": "f1",
                                           "caught_0.1pct": "c01", "warning_days": "w"})))
        lines += ["", "**Test period**, chosen before test was opened: the best LightGBM, the "
                  "best logistic regression and the best overall on validation. 3 seeds, "
                  "thresholds set on validation and applied unchanged. Benchmark rows first, "
                  "for comparison.", "",
                  "| Configuration | Caught at 1% false alarms | (actual false-alarm %) | "
                  "Caught at 0.1% false alarms | Median days of warning | PR-AUC |",
                  "|---|---|---|---|---|---|"]
        for name, g in rows:
            lines.append(f"| {name} | {pm(g['c1'])}% | {pm(g['f1'], digits=2)} | "
                         f"{pm(g['c01'])}% | {pm(g['w'], pct=False)} | "
                         f"{pm(g['pr_auc'], pct=False, digits=3)} |")
    return lines + [""]


def no_signal_section(n: int = 7) -> list[str]:
    """Failing test drives with none of the L1 error counters (E-4) above zero in
    their window, and how many of them each headline model catches at 1%."""
    from .features import FEATURES_DIR
    from .models import SMART_SIGNALS
    from .run import con, glob
    if not (FEATURES_DIR / "test").exists():
        return []
    any_signal = " OR ".join(f"coalesce({c}, 0) > 0" for c in SMART_SIGNALS)
    q = (f"SELECT serial_number, max(CASE WHEN {any_signal} THEN 1 ELSE 0 END) AS signal "
         f"FROM {glob('test')} WHERE days_to_failure BETWEEN 0 AND {n - 1} "
         f"AND days_to_failure <= days_to_period_end GROUP BY 1")
    sig = con().execute(q).df()
    quiet = sig.loc[sig["signal"] == 0, "serial_number"]
    lines = [f"## Failures with no error-counter signal ({n} days)", "",
             f"{len(quiet):,} of {len(sig):,} failing test drives "
             f"({len(quiet) / len(sig) * 100:.1f}%) have none of SMART "
             f"{', '.join(c[1:] for c in SMART_SIGNALS)} above zero on any day of their "
             f"{n}-day window. Share of them caught at the 1% false-alarm threshold set on "
             "validation (seed 0):", "",
             "| Model | Caught, no-signal drives | Caught, other failing drives |",
             "|---|---|---|"]
    for run_id, name in [("L1-M0-core-seed0", "SMART rule"),
                         ("L2-M0-core-seed0", "Logistic regression"),
                         ("L4-M0-core-seed0", "LightGBM")]:
        d = RESULTS / f"N{n}" / run_id
        if not (d / "test_drive_scores.parquet").exists():
            continue
        thr = json.loads((d / "metrics.json").read_text())["metrics"]["valid_threshold_1pct"][
            "threshold"]
        s = pd.read_parquet(d / "test_drive_scores.parquet")
        s = s[s["failing"]]
        hit = s["score"] >= thr
        q_mask = s["serial_number"].isin(quiet)
        lines.append(f"| {name} | {hit[q_mask].mean() * 100:.1f}% | "
                     f"{hit[~q_mask].mean() * 100:.1f}% |")
    return lines + [""]


def comparison(df: pd.DataFrame) -> str:
    """Resampling vs synthetic data on identical splits (E-3)."""
    d = df[(df["n"] == 7) & (df["features"] == "core") & df["level"].isin(["L2", "L4"])]
    lines = ["| Model | Class weighting (M0) | Undersampling (M1) | SMOTE (M2) | "
             "CTGAN synthetic failures (M3) |", "|---|---|---|---|---|"]
    for lv in ("L4", "L2"):
        cells = []
        for me in ("M0", "M1", "M2", "M3"):
            g = d[(d["level"] == lv) & (d["method"] == me)]
            cells.append(f"{pm(g['det_1pct'])}%" if len(g) else "not run")
        lines.append(f"| {LEVEL_NAMES[lv]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write() -> str:
    df = load()
    if df.empty:
        return "no results yet"
    df.to_csv(RESULTS / "all_runs.csv", index=False)
    acc = df.loc[df["n"] == 7, "accuracy_always_healthy"].dropna()
    parts = ["# Benchmark v1 results", "",
             "Generated by `dfb report` from `results/benchmark_v1/*/*/metrics.json`. "
             "Test period 2026 Q2–Q3; data hard drives; counted per drive "
             "(see `EVALUATION.md`). Values are mean ± standard deviation over seeds.", "",
             "**Thresholds are set on validation and applied unchanged to test** (decision "
             "E-2); the actual false-alarm rate that threshold gives on test is shown next to "
             "each detection rate. The test-curve column uses a threshold set on test itself, "
             "for comparison with results that set the threshold on test data.", "",
             "Each run's `metrics.json` records the commit it ran on (`code_version`). "
             "Those were development commits, later combined into this repository's single "
             "commit. What changed between them was documentation, memory use when scoring, "
             "and new optional settings for the studies, not how the existing runs train, "
             "score or evaluate.", ""]
    if len(acc):
        parts += [f"Why not accuracy: a model that always says \"healthy\" is right on "
                  f"{acc.iloc[0] * 100:.3f}% of test drive-days and catches no failures.", ""]
    parts += ["## Main result: warning window 7 days", "", table(df, 7), "",
              "## Resampling vs synthetic failures: first setup (7 days, core features)", "",
              "As first declared: rows added or removed without the training sampling weights. "
              "The corrected comparison is in the settings-study section below.", "",
              "Failures caught at 1% false alarms. M2 and M3 add exactly the same number of "
              "failure rows (decision E-3) on identical splits.", "", comparison(df), ""]
    for n in (14, 30):
        if (df["n"] == n).any():
            parts += [f"## Window: {n} days", "", table(df, n), ""]
    parts += smote_study_section()
    parts += improvement_section()
    parts += no_signal_section()
    parts += ["## Failures caught at other false-alarm budgets", "",
              "Mean over seeds. Here the threshold for each budget is set on the test scores "
              "themselves, so each budget is met exactly; the validation-threshold tables read 1–2 "
              "points higher because that threshold gives slightly more false alarms on test. It "
              "shows how detection grows as more false alarms are allowed. The SMART rule's "
              "score is a whole number, so its detection moves in steps.", "",
              budget_sweep(), ""]
    head = RESULTS / "N7" / f"{HEADLINE_RUN}-seed0"
    if (head / "slice_brand.csv").exists():
        parts += [f"## By slice: {LEVEL_NAMES['L4']}, class weighting, seed 0, threshold for 1% "
                  "false alarms", "",
                  "Detection rates only where a slice has at least 30 test failures "
                  "(decision R-3).", ""]
        for col, title in [("brand", "Brand"), ("capacity_tb", "Capacity (TB)"),
                           ("age_band", "Age (years)"), ("new_model", "Drive model unseen in "
                                                         "training (R-4)"),
                           ("model", "Drive model")]:
            s = pd.read_csv(head / f"slice_{col}.csv")
            s["detection_rate"] = (s["detection_rate"] * 100).round(1)
            s["false_alarm_rate"] = (s["false_alarm_rate"] * 100).round(2)
            s[col] = s[col].astype(object).where(s[col].notna(), "unknown")
            s["note"] = s["note"].fillna("")
            s["detection_rate"] = s["detection_rate"].astype(object).where(
                s["detection_rate"].notna(), "–")
            parts += [f"### {title}", "", s.rename(columns={
                "detection_rate": "caught %", "false_alarm_rate": "false alarms %"}).to_markdown(
                index=False, missingval="–"), ""]
        m = pd.read_csv(head / "by_month.csv")
        m["detection_rate"] = (m["detection_rate"] * 100).round(1)
        parts += ["### By month of failure (drift check)", "",
                  m.rename(columns={"detection_rate": "caught %"}).to_markdown(index=False), ""]
    text = "\n".join(parts)
    (RESULTS / "RESULTS.md").write_text(text)
    return text
