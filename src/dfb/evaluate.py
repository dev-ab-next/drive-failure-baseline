"""Benchmark metric code. Decisions B-1 to B-3, E-1, E-2.

The same functions score this project's models and anyone else's
predictions (`dfb evaluate`). Input is one score per drive per day plus the
label columns from the feature tables (`days_to_failure`,
`days_to_period_end`).

Per-drive rule (B-1, E-1), for warning window N:
- a failing drive fails inside the period; its score is its highest daily
  score in its window (failure day and the N-1 days before);
- a healthy drive does not fail inside the period; its score is its
  highest daily score over its usable days;
- alarms a failing drive raises before its window are not false alarms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .labels import usable

FAR_LEVELS = (0.001, 0.01)  # B-2
MIN_SLICE_FAILURES = 30     # R-3


@dataclass
class DriveScores:
    drives: pd.DataFrame   # one row per drive: serial_number, failing, score (+ attributes)
    window: pd.DataFrame   # failing drives' window rows: serial_number, days_to_failure, score


def drive_scores(rows: pd.DataFrame, n: int, attrs: list[str] | None = None) -> DriveScores:
    """Collapse daily scores to one score per drive (B-1, E-1)."""
    attrs = attrs or []
    r = rows[usable(rows, n)]
    dtf = r["days_to_failure"]
    fails_here = dtf.notna() & (dtf >= 0) & (dtf <= r["days_to_period_end"])
    failing_serials = set(r.loc[fails_here, "serial_number"])
    is_failing = r["serial_number"].isin(failing_serials)
    in_window = is_failing & dtf.between(0, n - 1)

    window = r.loc[in_window, ["serial_number", "days_to_failure", "score"]]
    fail = window.groupby("serial_number")["score"].max().rename("score").to_frame()
    fail["failing"] = True
    healthy = r[~is_failing].groupby("serial_number")["score"].max().rename("score").to_frame()
    healthy["failing"] = False
    drives = pd.concat([fail, healthy])
    if attrs:
        last = r.sort_values("date").groupby("serial_number")[attrs].last()
        drives = drives.join(last)
    return DriveScores(drives.reset_index(), window.reset_index(drop=True))


def threshold_for(healthy_scores: np.ndarray, far: float) -> float:
    """Lowest threshold t such that the share of healthy drives with score >= t
    is at most `far`. Returns +inf if even the top score is shared by more."""
    h = np.sort(np.asarray(healthy_scores, dtype=float))[::-1]
    if len(h) == 0:
        return float("inf")
    values, first_idx = np.unique(-h, return_index=True)  # descending unique values
    values = -values
    # number of healthy drives with score >= v is the index after the last copy of v
    counts_ge = np.searchsorted(-h, -values, side="right")
    ok = counts_ge / len(h) <= far
    if not ok.any():
        return float("inf")
    return float(values[ok][-1])


def detection(ds: DriveScores, thr: float) -> dict:
    d = ds.drives
    fail = d[d["failing"]]
    healthy = d[~d["failing"]]
    caught = fail["score"] >= thr
    alarms = ds.window[ds.window["score"] >= thr]
    warn = alarms.groupby("serial_number")["days_to_failure"].max()
    return {
        "threshold": thr,
        "detection_rate": float(caught.mean()) if len(fail) else float("nan"),
        "false_alarm_rate": float((healthy["score"] >= thr).mean()) if len(healthy) else float("nan"),
        "failures": int(len(fail)),
        "caught": int(caught.sum()),
        "healthy_drives": int(len(healthy)),
        "warning_days_median": float(warn.median()) if len(warn) else float("nan"),
        "warning_days_q1": float(warn.quantile(0.25)) if len(warn) else float("nan"),
        "warning_days_q3": float(warn.quantile(0.75)) if len(warn) else float("nan"),
    }


def summary(test: DriveScores, valid: DriveScores | None = None) -> dict:
    """Headline metrics. With `valid`, thresholds are also set on validation and
    applied unchanged to test (E-2, the headline); otherwise test-curve only."""
    d = test.drives
    y = d["failing"].astype(int).to_numpy()
    s = d["score"].to_numpy()
    out = {
        "pr_auc": float(average_precision_score(y, s)) if y.any() else float("nan"),
        "roc_auc": float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan"),
        "failures": int(y.sum()),
        "healthy_drives": int(len(y) - y.sum()),
    }
    healthy_test = d.loc[~d["failing"], "score"].to_numpy()
    for far in FAR_LEVELS:
        key = f"{far * 100:g}pct"
        out[f"test_curve_{key}"] = detection(test, threshold_for(healthy_test, far))
        if valid is not None:
            hv = valid.drives.loc[~valid.drives["failing"], "score"].to_numpy()
            out[f"valid_threshold_{key}"] = detection(test, threshold_for(hv, far))
    return out


def by_slice(test: DriveScores, thr: float, column: str,
             min_failures: int = MIN_SLICE_FAILURES) -> pd.DataFrame:
    """Detection and false-alarm rate per slice at one global threshold.
    Slices with fewer than `min_failures` failures show no rate (R-3)."""
    rows = []
    for value, g in test.drives.groupby(column, dropna=False, observed=True):
        fail = g[g["failing"]]
        healthy = g[~g["failing"]]
        enough = len(fail) >= min_failures
        rows.append({
            column: value,
            "failures": len(fail),
            "healthy_drives": len(healthy),
            "detection_rate": float((fail["score"] >= thr).mean()) if enough else None,
            "false_alarm_rate": float((healthy["score"] >= thr).mean()) if len(healthy) else None,
            "note": "" if enough else "too few to measure",
        })
    return pd.DataFrame(rows).sort_values("failures", ascending=False)


def by_month(test: DriveScores, rows: pd.DataFrame, thr: float) -> pd.DataFrame:
    """Detection at one threshold by month of failure (drift check)."""
    fail_day = rows.loc[rows["days_to_failure"] == 0, ["serial_number", "date"]]
    fail_day = fail_day.assign(month=pd.to_datetime(fail_day["date"]).dt.strftime("%Y-%m"))
    d = test.drives[test.drives["failing"]].merge(fail_day[["serial_number", "month"]],
                                                   on="serial_number", how="left")
    g = d.groupby("month").agg(failures=("score", "size"),
                               detection_rate=("score", lambda s: float((s >= thr).mean())))
    return g.reset_index()

