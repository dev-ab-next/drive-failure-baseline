"""Labels and the period-boundary rule, for any warning window N (L-1, S-4)."""

from __future__ import annotations

import pandas as pd

WINDOWS = (7, 14, 30)  # R-5: 7 days main; 14 and 30 days also reported


def label(df: pd.DataFrame, n: int) -> pd.Series:
    """1 if the drive fails within the next n days, counting the failure day
    (days_to_failure in 0..n-1), else 0. Decision L-1."""
    d = df["days_to_failure"]
    return (d.notna() & (d >= 0) & (d <= n - 1)).astype("int8")


def usable(df: pd.DataFrame, n: int) -> pd.Series:
    """True if the row's label is decided by days inside its own period, so
    it uses nothing from a later period (S-4, L-2):
    either the whole n-day window ends inside the period, or the drive
    fails inside the window *and* inside the period."""
    dte = df["days_to_period_end"]
    dtf = df["days_to_failure"]
    fails_inside = dtf.notna() & (dtf >= 0) & (dtf <= n - 1) & (dtf <= dte)
    return (dte >= n - 1) | fails_inside


def usable_sql(n: int) -> str:
    """The same rule as `usable`, for DuckDB queries."""
    return (f"(days_to_period_end >= {n - 1} OR (days_to_failure BETWEEN 0 AND {n - 1}"
            f" AND days_to_failure <= days_to_period_end))")
