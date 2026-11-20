import datetime as dt

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dfb import features
from dfb.labels import label, usable
from dfb.leakage import LeakageError, check
from dfb.schema import conform, load_declared

D0 = dt.date(2024, 1, 1)


def write_quarter(path, rows):
    """Rows: list of (date, serial, smart_5_raw, failure)"""
    t = pa.table({
        "date": pa.array([r[0] for r in rows], pa.date32()),
        "serial_number": [r[1] for r in rows],
        "model": ["ST12000NM0008"] * len(rows),
        "capacity_bytes": pa.array([12_000_000_000_000] * len(rows), pa.int64()),
        "failure": pa.array([r[3] for r in rows], pa.int8()),
        "smart_5_raw": pa.array([float(r[2]) for r in rows]),
        "smart_9_raw": pa.array([8760.0] * len(rows)),
    })
    pq.write_table(conform(t, load_declared()), path)


def run(tmp_path, rows, start, end, period_end, sample=False, fail=None):
    p = tmp_path / "q.parquet"
    write_quarter(p, rows)
    con = duckdb.connect()
    serials = sorted({r[1] for r in rows})
    drives = pd.DataFrame({"serial_number": serials, "model": "ST12000NM0008",
                           "brand": "Seagate", "kind": "data HDD", "capacity_tb": 12.0,
                           "first_fail_date": [fail.get(s) if fail else None for s in serials],
                           "drive_days": 1})
    drives["first_fail_date"] = pd.to_datetime(drives["first_fail_date"]).dt.date
    con.register("drives_df", pa.Table.from_pandas(drives, preserve_index=False))
    con.execute("CREATE TABLE drives AS SELECT * FROM drives_df")
    sql = features.feature_sql([str(p)], start, end, period_end, "train", sample)
    f = con.execute(sql).df()
    f["date"] = pd.to_datetime(f["date"]).dt.date
    return f


def days(n):
    return [D0 + dt.timedelta(days=i) for i in range(n)]


def test_delta_uses_value_from_seven_days_earlier(tmp_path):
    rows = [(d, "A", i, 0) for i, d in enumerate(days(20))]  # smart_5 grows by 1/day
    f = run(tmp_path, rows, D0, D0 + dt.timedelta(days=19), dt.date(2025, 9, 30))
    a = f.set_index("date")
    assert a.loc[D0 + dt.timedelta(days=10), "d7_5"] == 7
    assert a.loc[D0 + dt.timedelta(days=3), "d7_5"] == 3  # only 3 days of history


def test_features_never_use_future_days(tmp_path):
    base = [(d, "A", 0, 0) for d in days(20)]
    changed = [(d, "A", 999 if i >= 15 else 0, 0) for i, d in enumerate(days(20))]
    end = D0 + dt.timedelta(days=19)
    f1 = run(tmp_path, base, D0, end, dt.date(2025, 9, 30))
    f2 = run(tmp_path, changed, D0, end, dt.date(2025, 9, 30))
    cols = features.core_features()
    early = f1["date"] < D0 + dt.timedelta(days=15)
    pd.testing.assert_frame_equal(f1.loc[early, cols].reset_index(drop=True),
                                  f2.loc[early, cols].reset_index(drop=True))


def test_lookback_rows_are_not_output(tmp_path):
    rows = [(d, "A", i, 0) for i, d in enumerate(days(40))]
    start = D0 + dt.timedelta(days=35)
    f = run(tmp_path, rows, start, D0 + dt.timedelta(days=39), dt.date(2025, 9, 30))
    assert f["date"].min() == start
    assert f.loc[f["date"] == start, "d30_5"].iloc[0] == 30  # history came from lookback


def test_rows_after_failure_removed_and_days_to_failure(tmp_path):
    rows = [(d, "A", 0, 1 if i == 9 else 0) for i, d in enumerate(days(12))]
    fail = {"A": D0 + dt.timedelta(days=9)}
    f = run(tmp_path, rows, D0, D0 + dt.timedelta(days=11), dt.date(2025, 9, 30), fail=fail)
    assert f["date"].max() == D0 + dt.timedelta(days=9)
    assert f.set_index("date").loc[D0, "days_to_failure"] == 9


def test_training_sample_keeps_every_failure_window_day(tmp_path):
    rows = [(d, s, 0, 0) for s in "ABCDEFGHIJ" for d in days(60)]
    fail = {"A": D0 + dt.timedelta(days=59)}
    f = run(tmp_path, rows, D0, D0 + dt.timedelta(days=59), dt.date(2025, 9, 30),
            sample=True, fail=fail)
    a = f[f.serial_number == "A"]
    assert (a["days_to_failure"].between(0, 29)).sum() == 30
    neg = f[~f["days_to_failure"].between(0, 29).fillna(False)]
    assert (neg["sample_weight"] == 50).all()
    assert len(neg) < 0.1 * 570


@pytest.mark.parametrize("n", [7, 14, 30])
def test_label_and_boundary(n):
    df = pd.DataFrame({"days_to_failure": [0, n - 1, n, -1, None, None, 1, 5 + n],
                       "days_to_period_end": [n - 1, n - 2, 100, 100, 100, n - 2, 3, 2]})
    assert label(df, n).tolist() == [1, 1, 0, 0, 0, 0, 1, 0]
    # row 2: fails after the period ends -> label needs a later period -> unusable
    # row 6: healthy near the end -> unknown -> unusable
    # row 7: fails inside the window and the period -> usable
    # row 8: fails after the period ends -> unusable
    assert usable(df, n).tolist() == [True, False, True, True, True, False, True, False]


def frame(dates, period, dte):
    return pd.DataFrame({"date": dates, "period": period, "days_to_period_end": dte,
                         "days_to_failure": None})


def test_leakage_check_passes_and_fails():
    tr = frame([dt.date(2025, 9, 1)], "train", [29])
    te = frame([dt.date(2026, 4, 2)], "test", [100])
    assert check({"train": tr, "test": te}, 7)
    with pytest.raises(LeakageError, match="later period"):
        check({"train": frame([dt.date(2025, 9, 28)], "train", [2])}, 7)
    with pytest.raises(LeakageError, match="outside"):
        check({"train": frame([dt.date(2026, 4, 2)], "train", [100])}, 7)
    bad = te.assign(synthetic=True)
    with pytest.raises(LeakageError, match="synthetic"):
        check({"train": tr, "test": bad}, 7)
