"""Data window, split dates and paths (decisions D-2 and S-1 to S-4)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PARQUET_DIR = DATA_DIR / "parquet"
MANIFEST = DATA_DIR / "manifest.json"
REPORTS_DIR = ROOT / "reports"

BASE_URL = "https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data"

# Most recent 3 years = 12 quarters, (year, quarter).
QUARTERS: list[tuple[int, int]] = [
    (2023, 4),
    (2024, 1), (2024, 2), (2024, 3), (2024, 4),
    (2025, 1), (2025, 2), (2025, 3), (2025, 4),
    (2026, 1), (2026, 2), (2026, 3),
]

# Split by time. Each period is [start, end] inclusive.
TRAIN = (dt.date(2023, 10, 1), dt.date(2025, 9, 30))
VALID = (dt.date(2025, 10, 1), dt.date(2026, 3, 31))
TEST = (dt.date(2026, 4, 1), dt.date(2026, 9, 30))
PERIODS = {"train": TRAIN, "valid": VALID, "test": TEST}


def quarter_name(year: int, quarter: int) -> str:
    return f"Q{quarter}_{year}"


def quarter_url(year: int, quarter: int) -> str:
    return f"{BASE_URL}/data_{quarter_name(year, quarter)}.zip"


def period_of(day: dt.date) -> str | None:
    """Return 'train', 'valid', 'test', or None if outside the window."""
    for name, (start, end) in PERIODS.items():
        if start <= day <= end:
            return name
    return None
