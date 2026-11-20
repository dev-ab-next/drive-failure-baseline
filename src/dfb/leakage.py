"""Automated leakage check. A run stops if any rule is broken."""

from __future__ import annotations

import pandas as pd

from . import config
from .labels import usable


class LeakageError(AssertionError):
    pass


def check(frames: dict[str, pd.DataFrame], n: int) -> list[str]:
    """`frames` maps 'train' / 'valid' / 'test' to rows actually used.
    Returns the list of rules checked; raises LeakageError on a breach."""
    checked = []
    for name, df in frames.items():
        start, end = config.PERIODS[name]
        dates = pd.to_datetime(df["date"]).dt.date
        if len(df) and (dates.min() < start or dates.max() > end):
            raise LeakageError(f"{name}: dates {dates.min()}..{dates.max()} outside {start}..{end}")
        checked.append(f"{name}: all dates inside {start}..{end}")
        if (df["period"] != name).any():
            raise LeakageError(f"{name}: rows tagged with another period")
        bad = (~usable(df, n)).sum()
        if bad:
            raise LeakageError(f"{name}: {bad:,} rows whose {n}-day label depends on "
                               "a later period")
        checked.append(f"{name}: every {n}-day label is decided inside the period")
    order = [p for p in ("train", "valid", "test") if p in frames]
    for a, b in zip(order, order[1:]):
        if len(frames[a]) and len(frames[b]):
            if pd.to_datetime(frames[a]["date"]).max() >= pd.to_datetime(frames[b]["date"]).min():
                raise LeakageError(f"{a} overlaps {b} in time")
            checked.append(f"{a} ends before {b} starts")
    if "synthetic" in frames.get("train", pd.DataFrame()).columns:
        checked.append("synthetic rows exist only in train")
    for name in ("valid", "test"):
        if name in frames and "synthetic" in frames[name].columns and frames[name]["synthetic"].any():
            raise LeakageError(f"{name}: contains synthetic or resampled rows")
    return checked
