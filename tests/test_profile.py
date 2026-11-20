import datetime as dt

import pandas as pd
import pyarrow as pa
import pytest

from dfb import config
from dfb.profile import (aggregate_quarter, brand_of, bucket_of, drive_kind,
                         drives_from, is_ssd)


def day_table(rows):
    """Rows: (date, serial, model, failure)"""
    return pa.table({
        "date": pa.array([dt.date.fromisoformat(r[0]) for r in rows], pa.date32()),
        "serial_number": [r[1] for r in rows],
        "model": [r[2] for r in rows],
        "capacity_bytes": pa.array([4_000_000_000_000] * len(rows), pa.int64()),
        "failure": pa.array([r[3] for r in rows], pa.int8()),
        "smart_9_raw": pa.array([1000.0] * len(rows), pa.float64()),
    })


@pytest.mark.parametrize("day,period", [
    ("2023-10-01", "train"), ("2025-09-30", "train"),
    ("2025-10-01", "valid"), ("2026-03-31", "valid"),
    ("2026-04-01", "test"), ("2026-09-30", "test"),
    ("2023-09-30", None), ("2026-10-01", None),
])
def test_period_boundaries(day, period):
    assert config.period_of(dt.date.fromisoformat(day)) == period


@pytest.mark.parametrize("n,label", [
    (0, "0"), (1, "1-9"), (9, "1-9"), (10, "10-29"), (29, "10-29"),
    (30, "30-99"), (99, "30-99"), (100, "100-299"), (299, "100-299"),
    (300, "300+"), (5000, "300+"),
])
def test_bucket_edges(n, label):
    assert bucket_of(n) == label


def test_failed_drive_counted_once_with_first_flag_period():
    # Flagged on the last training day and again on the first validation day.
    q3 = aggregate_quarter(day_table([
        ("2025-09-29", "A", "ST4000DM000", 0),
        ("2025-09-30", "A", "ST4000DM000", 1),
        ("2025-09-30", "B", "ST4000DM000", 0),
    ]), "Q3_2025")
    q4 = aggregate_quarter(day_table([
        ("2025-10-01", "A", "ST4000DM000", 1),
        ("2025-10-01", "B", "ST4000DM000", 0),
    ]), "Q4_2025")
    drives = drives_from(pd.concat([q3, q4]))
    a = drives.set_index("serial_number").loc["A"]
    assert drives["failed"].sum() == 1
    assert a["fail_period"] == "train"
    assert a["drive_days"] == 3


def test_first_fail_date_with_quarters_without_failure():
    # A drive seen in a quarter without a failure, then failing in the next one.
    q3 = aggregate_quarter(day_table([
        ("2025-09-30", "A", "ST4000DM000", 0),
        ("2025-09-30", "B", "ST4000DM000", 0),
    ]), "Q3_2025")
    q4 = aggregate_quarter(day_table([
        ("2025-10-02", "A", "ST4000DM000", 1),
        ("2025-10-02", "B", "ST4000DM000", 0),
    ]), "Q4_2025")
    drives = drives_from(pd.concat([q3, q4])).set_index("serial_number")
    assert drives.loc["A", "first_fail_date"] == dt.date(2025, 10, 2)
    assert drives.loc["A", "fail_period"] == "valid"
    assert not drives.loc["B", "failed"]


def test_drive_model_is_most_reported_label():
    agg = aggregate_quarter(day_table([
        ("2024-01-01", "A", "ST8000NM0055", 0),
        ("2024-01-02", "A", "ST8000NM0055", 0),
        ("2024-01-03", "A", "ST8000NM005", 0),
    ]), "Q1_2024")
    drives = drives_from(agg)
    assert len(drives) == 1
    assert drives.iloc[0]["model"] == "ST8000NM0055"


@pytest.mark.parametrize("model,brand", [
    ("ST12000NM0008", "Seagate"),
    ("Seagate BarraCuda SSD ZA250CM10002", "Seagate"),
    ("HGST HUH721212ALN604", "HGST"),
    ("WDC WUH721816ALE6L4", "Western Digital"),
    ("WUH722222ALE6L4", "Western Digital"),
    ("TOSHIBA MG07ACA14TA", "Toshiba"),
    ("MTFDDAV240TCB", "Micron"),
    ("CT250MX500SSD1", "Crucial"),
    ("DELLBOSS VD", "Dell BOSS"),
    ("Hitachi HDS5C4040ALE630", "HGST"),
    ("SSDSCKKB240GZR", "Intel"),
    ("SAMSUNG MZ7L3480HCHQ", "Samsung"),
    ("Samsung SSD 870 EVO 2TB", "Samsung"),
    ("SOMETHING ELSE", "Other"),
])
def test_brand(model, brand):
    assert brand_of(model) == brand


@pytest.mark.parametrize("model,ssd", [
    ("Seagate BarraCuda SSD ZA250CM10002", True),
    ("MTFDDAV240TCB", True),
    ("CT250MX500SSD1", True),
    ("DELLBOSS VD", True),
    ("SSDSCKKB240GZR", True),
    ("Micron 5300 MTFDDAK480TDS", True),
    ("SAMSUNG MZ7L3480HCHQ", True),
    ("WD Blue SA510 2.5 250GB", True),
    ("WDC WDS250G2B0A", True),
    ("Seagate IronWolf ZA250NM10002", True),
    ("WDC WD5000LPVX", False),
    ("TOSHIBA MQ01ABF050", False),
    ("ST12000NM0008", False),
    ("TOSHIBA MG07ACA14TA", False),
])
def test_ssd(model, ssd):
    assert is_ssd(model) == ssd


@pytest.mark.parametrize("model,tb,kind", [
    ("ST12000NM0008", 12.0, "data HDD"),
    ("TOSHIBA MQ01ABF050", 0.5, "boot HDD"),
    ("ST500LM030", 0.5, "boot HDD"),
    ("CT250MX500SSD1", 0.25, "boot SSD"),
    ("Samsung SSD 870 EVO 2TB", 2.0, "boot SSD"),
])
def test_drive_kind(model, tb, kind):
    assert drive_kind(model, tb) == kind
