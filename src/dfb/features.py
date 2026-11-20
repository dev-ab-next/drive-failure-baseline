"""Drive table, features, labels and split columns.

Output, one Parquet file per quarter:
  data/features/train/<Q>/part-*.parquet   every failure-window day + a fixed
                                           2% sample of other days, with weights
  data/features/valid/<Q>/part-*.parquet   every day
  data/features/test/<Q>/part-*.parquet    every day

Every row carries `days_to_failure` and `days_to_period_end`, so labels and
the period-boundary rule can be applied for any warning window N
(see `labels.py` and `leakage.py`). Decisions: R-1, S-4, F-9 to F-12, L-1.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import duckdb
import pandas as pd

from . import config
from .profile import brand_of, drive_kind, drives_from, load_aggregates
from .schema import check_file_schema

FEATURES_DIR = config.DATA_DIR / "features"
DRIVES_FILE = config.DATA_DIR / "derived" / "drives.parquet"

# F-9: columns every brand reports (~93-100% of data-HDD rows).
CORE = [1, 3, 4, 5, 7, 9, 10, 12, 192, 193, 194, 197, 198, 199]
# F-10: error counters whose recent change is a feature.
DELTA = [5, 187, 188, 197, 198, 199]
DELTA_DAYS = [7, 30]
# F-11: brand-specific columns, used only in the F-full feature set.
EXTRA = [2, 8, 18, 22, 23, 24, 27, 71, 90, 184, 187, 188, 189, 190, 191, 195,
         196, 200, 220, 222, 223, 224, 226, 240, 241, 242]
BRANDS = ["Seagate", "Toshiba", "HGST", "Western Digital"]
LOOKBACK_DAYS = max(DELTA_DAYS)
# F-12: negative sample for training. Kept if hash(serial, date, salt) falls
# under SAMPLE_RATE; positives within MAX_WINDOW days of failure always kept.
SAMPLE_RATE = 0.02
SAMPLE_SALT = "dfb-benchmark-v1"
MAX_WINDOW = 30


def raw(n: int) -> str:
    return f"smart_{n}_raw"


def core_features() -> list[str]:
    cols = [f"s{n}" for n in CORE]
    cols += [f"d{k}_{n}" for n in DELTA for k in DELTA_DAYS]
    cols += ["age_years", "capacity_tb"] + [f"brand_{b.split()[0].lower()}" for b in BRANDS]
    return cols


def full_features() -> list[str]:
    return core_features() + [f"s{n}" for n in EXTRA if n not in CORE]


# ------------------------------------------------------------ drive table

def build_drive_table(path: Path = DRIVES_FILE) -> pd.DataFrame:
    """Serial -> model, brand, kind, capacity, first failure date (R-1, F-2..F-7)."""
    agg = load_aggregates()
    d = drives_from(agg)
    d["capacity_tb"] = (d["capacity_bytes"] / 1e12).round(1)
    d["brand"] = d["model"].map(brand_of)
    d["kind"] = [drive_kind(m, c) for m, c in zip(d["model"], d["capacity_tb"])]
    out = d[["serial_number", "model", "brand", "kind", "capacity_tb",
             "first_fail_date", "drive_days"]].copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out


# --------------------------------------------------------------- features

def quarter_files() -> dict[str, Path]:
    return {config.quarter_name(y, q): config.PARQUET_DIR / f"{config.quarter_name(y, q)}.parquet"
            for y, q in config.QUARTERS}


def quarter_dates(name: str) -> tuple[dt.date, dt.date]:
    from .schema import quarter_bounds
    return quarter_bounds(name)


BUCKETS = 8  # drives are split into groups by serial hash to bound memory


def feature_sql(files: list[str], start: dt.date, end: dt.date, period_end: dt.date,
                period: str, sample: bool, bucket: int | None = None,
                buckets: int = BUCKETS) -> str:
    """SQL for one chunk of days [start, end], reading LOOKBACK_DAYS earlier.
    With `bucket`, only drives whose serial hash falls in that group; a
    drive's whole history is always in one group, so results do not change."""
    bucket_clause = ("" if bucket is None else
                     f"AND hash(r.serial_number || '|bucket') % {buckets} = {bucket}")
    look = start - dt.timedelta(days=LOOKBACK_DAYS)
    smart_cols = sorted(set(CORE) | set(DELTA) | set(EXTRA))
    select_src = ", ".join(f"r.{raw(n)}" for n in smart_cols)
    windows = []
    for n in DELTA:
        for k in DELTA_DAYS:
            windows.append(
                f"{raw(n)} - first_value({raw(n)} IGNORE NULLS) OVER w{k} AS d{k}_{n}")
    keep_raw = ", ".join(f"{raw(n)} AS s{n}" for n in smart_cols)
    brand_cols = ", ".join(
        f"CAST(d.brand = '{b}' AS FLOAT) AS brand_{b.split()[0].lower()}" for b in BRANDS)
    feature_cols = ", ".join(f"CAST(f.s{n} AS FLOAT) AS s{n}" for n in smart_cols)
    delta_cols = ", ".join(f"CAST(f.d{k}_{n} AS FLOAT) AS d{k}_{n}"
                           for n in DELTA for k in DELTA_DAYS)
    file_list = ", ".join(f"'{f}'" for f in files)
    sample_clause = ""
    weight = "1.0"
    if sample:
        thresh = int(SAMPLE_RATE * 1_000_000)
        keep = (f"(hash(f.serial_number || '|' || CAST(f.date AS VARCHAR) || '|{SAMPLE_SALT}')"
                f" % 1000000) < {thresh}")
        positive = f"(datediff('day', f.date, d.first_fail_date) BETWEEN 0 AND {MAX_WINDOW - 1})"
        sample_clause = f"AND ({positive} OR {keep})"
        weight = f"CASE WHEN {positive} THEN 1.0 ELSE {1 / SAMPLE_RATE} END"
    return f"""
    WITH src AS (
        SELECT r.date, r.serial_number, {select_src}
        FROM read_parquet([{file_list}]) r
        JOIN drives d ON d.serial_number = r.serial_number
        WHERE d.kind = 'data HDD'
          AND r.date BETWEEN DATE '{look}' AND DATE '{end}'
          {bucket_clause}
    ),
    feat AS (
        SELECT date, serial_number, {keep_raw}, {", ".join(windows)}
        FROM src
        WINDOW {", ".join(
            f"w{k} AS (PARTITION BY serial_number ORDER BY date "
            f"RANGE BETWEEN INTERVAL {k} DAYS PRECEDING AND CURRENT ROW)"
            for k in DELTA_DAYS)}
    )
    SELECT f.date, f.serial_number, d.model, d.brand,
           {feature_cols}, {delta_cols},
           CAST(f.s9 / 8760.0 AS FLOAT) AS age_years,
           CAST(d.capacity_tb AS FLOAT) AS capacity_tb,
           {brand_cols},
           datediff('day', f.date, d.first_fail_date) AS days_to_failure,
           datediff('day', f.date, DATE '{period_end}') AS days_to_period_end,
           '{period}' AS period,
           CAST({weight} AS FLOAT) AS sample_weight
    FROM feat f
    JOIN drives d ON d.serial_number = f.serial_number
    WHERE f.date BETWEEN DATE '{start}' AND DATE '{end}'
      AND (d.first_fail_date IS NULL OR f.date <= d.first_fail_date)
      {sample_clause}
    """


def connect(memory_limit: str = "6GB", threads: int = 3) -> duckdb.DuckDBPyConnection:
    tmp = config.DATA_DIR / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.execute("SET enable_progress_bar=false")
    except duckdb.InvalidInputException:  # inside Jupyter without ipywidgets
        pass
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET threads={threads}")
    con.execute(f"SET temp_directory='{tmp}'")
    con.execute("SET preserve_insertion_order=false")
    return con


def build_quarter(con, name: str, files: dict[str, Path], out_dir: Path = FEATURES_DIR) -> Path:
    start, end = quarter_dates(name)
    period = config.period_of(start)
    if period is None:
        raise ValueError(name)
    period_end = config.PERIODS[period][1]
    names = list(files)
    i = names.index(name)
    use = [files[name]] + ([files[names[i - 1]]] if i > 0 else [])
    for f in use:
        check_file_schema(f)
    out = out_dir / period / name
    out.mkdir(parents=True, exist_ok=True)
    for b in range(BUCKETS):
        sql = feature_sql([str(f) for f in use], start, end, period_end, period,
                          sample=(period == "train"), bucket=b)
        part = out / f"part-{b}.parquet"
        con.execute(f"COPY ({sql}) TO '{part}' (FORMAT parquet, COMPRESSION zstd)")
    return out


def build_all(quarters: list[str] | None = None) -> None:
    if not DRIVES_FILE.exists():
        print("building drive table")
        build_drive_table()
    files = quarter_files()
    con = connect()
    con.execute(f"CREATE TABLE drives AS SELECT * FROM read_parquet('{DRIVES_FILE}')")
    for name in quarters or list(files):
        out = build_quarter(con, name, files)
        n = con.execute(f"SELECT count(*) FROM read_parquet('{out}/*.parquet')").fetchone()[0]
        print(f"{name}: {n:,} rows -> {out.relative_to(config.ROOT)}")
