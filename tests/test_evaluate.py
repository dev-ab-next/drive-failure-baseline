import datetime as dt

import numpy as np
import pandas as pd
import pytest

from dfb.evaluate import (by_slice, detection, drive_scores, summary, threshold_for)

D = dt.date(2026, 5, 1)


def rows(spec):
    """spec: list of (serial, day_offset, score, days_to_failure or None)"""
    return pd.DataFrame({
        "serial_number": [s[0] for s in spec],
        "date": [D + dt.timedelta(days=s[1]) for s in spec],
        "score": [s[2] for s in spec],
        "days_to_failure": [s[3] for s in spec],
        "days_to_period_end": [100 - s[1] for s in spec],
        "brand": ["Seagate"] * len(spec),
    })


def test_failing_drive_uses_only_its_window_and_early_alarm_is_ignored():
    # F fails on day 10. A high score on day 0 (10 days before) is outside a
    # 7-day window: it neither catches the failure nor counts as a false alarm.
    r = rows([("F", 0, 0.9, 10), ("F", 5, 0.2, 5), ("F", 10, 0.3, 0),
              ("H", 0, 0.1, None), ("H", 1, 0.5, None)])
    ds = drive_scores(r, 7)
    d = ds.drives.set_index("serial_number")
    assert d.loc["F", "failing"] and d.loc["F", "score"] == 0.3
    assert not d.loc["H", "failing"] and d.loc["H", "score"] == 0.5


def test_drive_failing_after_period_is_not_a_failure_here():
    r = rows([("A", 0, 0.9, 150)])  # fails 150 days later, after the period ends
    d = drive_scores(r, 7).drives
    assert not d.iloc[0]["failing"]


@pytest.mark.parametrize("scores,far,expected", [
    ([0.1] * 99 + [0.9], 0.01, 0.9),     # exactly 1% at or above 0.9
    ([0.1] * 98 + [0.9, 0.9], 0.01, float("inf")),  # tie: 2% share the top value
    ([0, 0, 0, 1], 0.5, 1.0),
    ([5, 4, 3, 2, 1, 0, 0, 0, 0, 0], 0.3, 3.0),
])
def test_threshold_respects_false_alarm_budget(scores, far, expected):
    t = threshold_for(np.array(scores, dtype=float), far)
    assert t == expected
    if np.isfinite(t):
        assert (np.array(scores) >= t).mean() <= far


def test_detection_and_warning_days():
    r = rows([("F1", 0, 0.8, 6), ("F1", 6, 0.2, 0),       # alarm 6 days before failure
              ("F2", 0, 0.1, 3), ("F2", 3, 0.1, 0),       # never alarms
              ] + [(f"H{i}", 0, 0.1, None) for i in range(98)] + [("H98", 0, 0.9, None)])
    ds = drive_scores(r, 7)
    out = detection(ds, 0.5)
    assert out["detection_rate"] == 0.5 and out["caught"] == 1
    assert out["false_alarm_rate"] == pytest.approx(1 / 99)
    assert out["warning_days_median"] == 6


def test_summary_uses_validation_threshold_on_test():
    test = drive_scores(rows([("F", 0, 0.6, 0)] + [(f"H{i}", 0, i / 100, None) for i in range(100)]), 7)
    valid = drive_scores(rows([(f"V{i}", 0, 0.5 * i / 100, None) for i in range(100)]), 7)
    s = summary(test, valid)
    v = s["valid_threshold_1pct"]
    assert v["threshold"] == pytest.approx(0.495)
    assert v["detection_rate"] == 1.0
    assert v["false_alarm_rate"] > 0.01  # validation threshold is too low for test: reported, not hidden
    assert s["test_curve_1pct"]["false_alarm_rate"] <= 0.01


def test_slices_mark_too_few():
    r = rows([(f"F{i}", 0, 0.9, 0) for i in range(5)] + [("H", 0, 0.1, None)])
    t = by_slice(drive_scores(r, 7, attrs=["brand"]), 0.5, "brand")
    assert t.iloc[0]["note"] == "too few to measure" and t.iloc[0]["detection_rate"] is None
