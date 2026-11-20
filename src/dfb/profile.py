"""Failure profile per drive model, brand, capacity, age and over time.

Reads the per-quarter Parquet files written by `dfb prepare` and writes
`reports/failure_profile.md` plus CSVs in `reports/`.

Definitions
- A drive is one serial number. Its model is the label it reports on the
  most days.
- A failed drive has `failure == 1` on at least one day; it is counted
  once, and its failure date is the first flagged day.
- AFR (annualized failure rate) = failures / (drive-days / 365) x 100%,
  the formula Backblaze uses in its own reports.
"""

from __future__ import annotations

import datetime as dt
import re

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import config
from .schema import check_file_schema

BUCKETS = [(0, 0, "0"), (1, 9, "1-9"), (10, 29, "10-29"), (30, 99, "30-99"),
           (100, 299, "100-299"), (300, None, "300+")]
CUTOFFS = [10, 30, 100, 300]
WINDOW_MIN_FAILURES = 10
AGE_BANDS = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 7), (7, None)]


# ---------------------------------------------------------------- helpers

def bucket_of(n: int) -> str:
    for low, high, label in BUCKETS:
        if n >= low and (high is None or n <= high):
            return label
    raise ValueError(n)


def brand_of(model: str) -> str:
    m = model.strip().upper()
    if m.startswith("ST") or m.startswith("SEAGATE"):
        return "Seagate"
    if m.startswith(("HGST", "HITACHI")):
        return "HGST"
    if m.startswith(("WDC", "WD", "WU")):
        return "Western Digital"
    if m.startswith(("TOSHIBA", "MG")):
        return "Toshiba"
    if m.startswith(("MTFD", "MICRON")):
        return "Micron"
    if m.startswith("CT"):
        return "Crucial"
    if m.startswith("DELLBOSS"):
        return "Dell BOSS"
    if m.startswith("SSDSC"):
        return "Intel"
    if m.startswith(("SAMSUNG", "MZ")):
        return "Samsung"
    return "Other"


SSD_PREFIXES = ("MTFD", "MICRON", "CT", "DELLBOSS", "SSDSC", "SAMSUNG MZ", "MZ")
SSD_PATTERNS = re.compile(r"SSD|SA510|\bWDS\d|\bZA\d")
BOOT_MAX_TB = 2.0  # drives under 2 TB are boot drives in this fleet (decision F-7)


def is_ssd(model: str) -> bool:
    m = model.strip().upper()
    return m.startswith(SSD_PREFIXES) or bool(SSD_PATTERNS.search(m))


def drive_kind(model: str, capacity_tb: float) -> str:
    """'boot SSD', 'boot HDD' or 'data HDD' (decision F-7)."""
    if is_ssd(model):
        return "boot SSD"
    if pd.notna(capacity_tb) and capacity_tb < BOOT_MAX_TB:
        return "boot HDD"
    return "data HDD"


def quarter_of(day: dt.date) -> str:
    return config.quarter_name(day.year, (day.month - 1) // 3 + 1)


def quarter_start(name: str) -> dt.date:
    q, year = name[1], int(name[3:])
    return dt.date(year, 3 * (int(q) - 1) + 1, 1)


def afr(failures, drive_days):
    if isinstance(drive_days, pd.Series):
        return (failures / (drive_days / 365) * 100).where(drive_days > 0)
    return failures / (drive_days / 365) * 100 if drive_days > 0 else float("nan")


# ----------------------------------------------------------- aggregation

def aggregate_quarter(table: pa.Table, quarter: str) -> pd.DataFrame:
    """One row per (serial, model label) in one quarter."""
    has_age = "smart_9_raw" in table.column_names
    cols = ["serial_number", "model", "date", "failure", "capacity_bytes"]
    t = table.select(cols + (["smart_9_raw"] if has_age else []))
    aggs = [("date", "count"), ("failure", "max"), ("capacity_bytes", "max")]
    if has_age:
        aggs.append(("smart_9_raw", "max"))
    g = t.group_by(["serial_number", "model"]).aggregate(aggs).to_pandas()
    g = g.rename(columns={"date_count": "days", "failure_max": "flagged",
                          "capacity_bytes_max": "capacity_bytes",
                          "smart_9_raw_max": "age_hours"})
    if "age_hours" not in g:
        g["age_hours"] = float("nan")
    fails = t.filter(pc.equal(t["failure"], 1))
    first = (fails.group_by(["serial_number"]).aggregate([("date", "min")])
             .to_pandas().rename(columns={"date_min": "first_fail_date"}))
    g = g.merge(first, on="serial_number", how="left")
    g["quarter"] = quarter
    return g


def drives_from(agg: pd.DataFrame) -> pd.DataFrame:
    """One row per serial number."""
    by_label = agg.groupby(["serial_number", "model"], as_index=False)["days"].sum()
    by_label = by_label.sort_values(["serial_number", "days", "model"],
                                    ascending=[True, False, True])
    model = by_label.drop_duplicates("serial_number").set_index("serial_number")["model"]
    d = agg.groupby("serial_number").agg(
        drive_days=("days", "sum"),
        capacity_bytes=("capacity_bytes", "max"),
        age_hours=("age_hours", "max"),
        n_quarters=("quarter", "nunique"),
    )
    # Earliest failure date over the quarters in which the drive failed. Dates and
    # empty values cannot be compared, so quarters without a failure are left out.
    failed = agg.dropna(subset=["first_fail_date"])
    d["first_fail_date"] = failed.groupby("serial_number")["first_fail_date"].min()
    d["model"] = model
    d["failed"] = d["first_fail_date"].notna()
    d["fail_period"] = d["first_fail_date"].map(
        lambda x: config.period_of(x) if pd.notna(x) else None)
    d["fail_quarter"] = d["first_fail_date"].map(
        lambda x: quarter_of(x) if pd.notna(x) else None)
    return d.reset_index()


# --------------------------------------------------------------- tables

def load_aggregates() -> pd.DataFrame:
    parts = []
    for year, q in config.QUARTERS:
        name = config.quarter_name(year, q)
        path = config.PARQUET_DIR / f"{name}.parquet"
        if not path.exists():
            print(f"{name}: missing, skipped")
            continue
        check_file_schema(path)  # refuse files not in the declared schema
        names = pq.read_schema(path).names
        cols = ["date", "serial_number", "model", "capacity_bytes", "failure"]
        cols += [c for c in ["smart_9_raw"] if c in names]
        print(f"{name}: aggregating")
        parts.append(aggregate_quarter(pq.read_table(path, columns=cols), name))
    if not parts:
        raise SystemExit("no Parquet files found; run `dfb prepare` first")
    agg = pd.concat(parts, ignore_index=True)
    agg["period"] = agg["quarter"].map(lambda n: config.period_of(quarter_start(n)))
    return agg


def model_table(drives: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
    canon = drives.set_index("serial_number")["model"]
    a = agg.assign(model=agg["serial_number"].map(canon))
    days = a.pivot_table(index="model", columns="period", values="days",
                         aggfunc="sum", fill_value=0)
    test_drives = a[a["period"] == "test"].groupby("model")["serial_number"].nunique()
    quarters = a.groupby("model")["quarter"].agg(["nunique", "min", "max"])
    order = {config.quarter_name(y, q): i for i, (y, q) in enumerate(config.QUARTERS)}
    first_q = a.assign(o=a["quarter"].map(order)).groupby("model")["o"].agg(["min", "max"])
    names = {v: k for k, v in order.items()}

    m = drives.groupby("model").agg(
        drives=("serial_number", "count"),
        drive_days=("drive_days", "sum"),
        failures=("failed", "sum"),
        capacity_tb=("capacity_bytes", lambda s: round(s.median() / 1e12, 1)),
    )
    for p in ["train", "valid", "test"]:
        m[f"failures_{p}"] = (drives[drives["fail_period"] == p]
                              .groupby("model").size().reindex(m.index, fill_value=0))
        m[f"drive_days_{p}"] = days.get(p, pd.Series(dtype=int)).reindex(m.index, fill_value=0)
    m["afr_pct"] = afr(m["failures"], m["drive_days"]).round(2)
    m["drives_in_test"] = test_drives.reindex(m.index, fill_value=0)
    m["quarters_present"] = quarters["nunique"]
    m["first_quarter"] = first_q["min"].map(names)
    m["last_quarter"] = first_q["max"].map(names)
    m["brand"] = [brand_of(x) for x in m.index]
    m["kind"] = [drive_kind(x, c) for x, c in zip(m.index, m["capacity_tb"])]
    m = m.sort_values("failures", ascending=False)
    lead = ["brand", "kind", "capacity_tb", "drives", "drive_days", "failures",
            "afr_pct", "failures_train", "failures_valid", "failures_test",
            "drives_in_test", "quarters_present", "first_quarter", "last_quarter"]
    return m[lead + [c for c in m.columns if c not in lead]]


def bucket_table(models: pd.DataFrame, column: str) -> pd.DataFrame:
    b = models.assign(bucket=models[column].map(bucket_of))
    t = b.groupby("bucket").agg(models=("drives", "size"), drives=("drives", "sum"),
                                failures=("failures", "sum"))
    t = t.reindex([lbl for _, _, lbl in BUCKETS], fill_value=0)
    t["share_of_drives_pct"] = (t["drives"] / t["drives"].sum() * 100).round(1)
    t["share_of_failures_pct"] = (t["failures"] / t["failures"].sum() * 100).round(1)
    t.index.name = f"{column} bucket"
    return t


def coverage_table(models: pd.DataFrame) -> pd.DataFrame:
    rows = []
    tot_d, tot_f = models["drives"].sum(), models["failures"].sum()
    tot_ft = models["failures_test"].sum()
    for c in CUTOFFS:
        k = models[models["failures"] >= c]
        kt = models[models["failures_test"] >= c]
        rows.append({
            "cutoff (failures)": c,
            "models kept (all-period failures >= cutoff)": len(k),
            "share of drives pct": round(k["drives"].sum() / tot_d * 100, 1),
            "share of failures pct": round(k["failures"].sum() / tot_f * 100, 1),
            "models with test failures >= cutoff": len(kt),
            "share of test failures pct": round(kt["failures_test"].sum() / tot_ft * 100, 1)
            if tot_ft else float("nan"),
        })
    return pd.DataFrame(rows).set_index("cutoff (failures)")


def group_table(models: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    t = models.groupby(by).agg(models=("drives", "size"), drives=("drives", "sum"),
                               drive_days=("drive_days", "sum"),
                               failures=("failures", "sum"))
    t["afr_pct"] = afr(t["failures"], t["drive_days"]).round(2)
    t["share_of_failures_pct"] = (t["failures"] / t["failures"].sum() * 100).round(1)
    return t.sort_values("failures", ascending=False)


def age_table(drives: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
    """AFR by drive age (power-on hours, SMART 9), data HDDs only.

    Approximation: each serial-quarter is placed in one age band using its
    highest power-on hours that quarter; a failure is placed in the band of
    the quarter it happened in.
    """
    cap_tb = drives["capacity_bytes"] / 1e12
    kinds = [drive_kind(m, c) for m, c in zip(drives["model"], cap_tb)]
    hdd = set(drives.loc[[k == "data HDD" for k in kinds], "serial_number"])
    fq = drives.set_index("serial_number")["fail_quarter"]
    a = agg[agg["serial_number"].isin(hdd) & agg["age_hours"].notna()].copy()
    a = a.drop_duplicates(["serial_number", "quarter"])
    a["years"] = a["age_hours"] / 8760
    a["failed_here"] = a["serial_number"].map(fq) == a["quarter"]

    def band(y):
        for lo, hi in AGE_BANDS:
            if y >= lo and (hi is None or y < hi):
                return f"{lo}-{hi} yrs" if hi else f"{lo}+ yrs"
        return "unknown"

    a["age_band"] = a["years"].map(band)
    t = a.groupby("age_band").agg(drive_days=("days", "sum"), failures=("failed_here", "sum"))
    t = t.reindex([band(lo) for lo, _ in AGE_BANDS]).fillna(0).astype(int)
    t["afr_pct"] = afr(t["failures"], t["drive_days"]).round(2)
    t["share_of_failures_pct"] = (t["failures"] / t["failures"].sum() * 100).round(1)
    return t


def fleet_table(drives: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
    canon = drives.set_index("serial_number")["model"]
    a = agg.assign(model=agg["serial_number"].map(canon))
    t = a.drop_duplicates(["serial_number", "quarter"]).pivot_table(
        index="model", columns="quarter", values="serial_number",
        aggfunc="count", fill_value=0)
    cols = [config.quarter_name(y, q) for y, q in config.QUARTERS]
    return t.reindex(columns=[c for c in cols if c in t.columns], fill_value=0)


def window_table(drives: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
    """Would a shorter training window lose drive models seen in test?"""
    train_q = [config.quarter_name(y, q) for y, q in config.QUARTERS
               if config.period_of(dt.date(y, 3 * (q - 1) + 1, 1)) == "train"]
    canon = drives.set_index("serial_number")["model"]
    a = agg.assign(model=agg["serial_number"].map(canon))
    test_serials = a.loc[a["period"] == "test", "serial_number"].unique()
    test = drives[drives["serial_number"].isin(test_serials)]
    test_fail = drives[drives["fail_period"] == "test"]
    rows = []
    for n in range(2, len(train_q) + 1, 2):
        window = train_q[-n:]
        in_w = a[a["quarter"].isin(window)]
        seen = set(in_w["model"])
        wf = drives[drives["fail_quarter"].isin(window)].groupby("model").size()
        strong = set(wf[wf >= WINDOW_MIN_FAILURES].index)
        rows.append({
            "training window": f"{window[0]} to {window[-1]} ({n / 4:g} yr)",
            "training drive-days": int(in_w["days"].sum()),
            "training failures": int(wf.sum()),
            f"models with >= {WINDOW_MIN_FAILURES} training failures": len(strong),
            "test drives whose model is seen in training pct":
                round(test["model"].isin(seen).mean() * 100, 1),
            f"test drives whose model has >= {WINDOW_MIN_FAILURES} training failures pct":
                round(test["model"].isin(strong).mean() * 100, 1),
            f"test failures whose model has >= {WINDOW_MIN_FAILURES} training failures pct":
                round(test_fail["model"].isin(strong).mean() * 100, 1)
                if len(test_fail) else float("nan"),
        })
    return pd.DataFrame(rows).set_index("training window")


def concentration(models: pd.DataFrame) -> pd.DataFrame:
    f = models["failures"].sort_values(ascending=False)
    d = models.loc[f.index, "drives"]
    rows = []
    for k in [1, 3, 5, 10, 20]:
        rows.append({"top models by failures": k,
                     "share of failures pct": round(f.iloc[:k].sum() / f.sum() * 100, 1),
                     "share of drives pct": round(d.iloc[:k].sum() / d.sum() * 100, 1)})
    return pd.DataFrame(rows).set_index("top models by failures")


# --------------------------------------------------------------- report

def md(df: pd.DataFrame) -> str:
    """Markdown table: integers with thousands separators, floats to 2 places."""
    out = df.copy()
    if isinstance(out.index, pd.MultiIndex):
        out = out.reset_index()
    for c in out.columns:
        col = out[c]
        if pd.api.types.is_bool_dtype(col):
            continue
        if pd.api.types.is_integer_dtype(col):
            out[c] = col.map(lambda x: f"{x:,}")
        elif pd.api.types.is_float_dtype(col):
            out[c] = col.map(lambda x: "" if pd.isna(x) else f"{x:,.2f}")
    return out.to_markdown(index=not isinstance(df.index, pd.MultiIndex),
                           disable_numparse=True)


def run() -> None:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    agg = load_aggregates()
    drives = drives_from(agg)
    models = model_table(drives, agg)
    fleet = fleet_table(drives, agg)

    tables = {
        "models": models,
        "buckets_all": bucket_table(models, "failures"),
        "buckets_test": bucket_table(models, "failures_test"),
        "coverage": coverage_table(models),
        "brands": group_table(models, ["brand", "kind"]),
        "capacity": group_table(models[models["kind"] == "data HDD"], ["capacity_tb"])
        .sort_index(),
        "age": age_table(drives, agg),
        "window": window_table(drives, agg),
        "concentration": concentration(models),
        "fleet_by_quarter": fleet,
    }
    # Machine-readable copies of the tables the notebook reads; the rest are in the report.
    for name in ("models", "brands", "age", "window"):
        tables[name].to_csv(config.REPORTS_DIR / f"profile_{name}.csv")

    quarters = sorted(agg["quarter"].unique(), key=quarter_start)
    unmatched = models[models["brand"] == "Other"].index.tolist()
    top = models.head(25)
    top_fleet = fleet.loc[models.sort_values("drive_days", ascending=False).index[:25]]

    out = [
        "# Failure profile: Backblaze Drive Stats",
        "",
        f"Generated by `dfb profile` on {dt.date.today().isoformat()}. "
        "Data: Backblaze Drive Stats (https://www.backblaze.com/cloud-storage/"
        "resources/hard-drive-test-data), credited per Backblaze's terms.",
        "",
        "## Summary",
        "",
        f"- Quarters: {len(quarters)} ({quarters[0]} to {quarters[-1]})",
        f"- Drive-days (daily records): {int(agg['days'].sum()):,}",
        f"- Drives (serial numbers): {len(drives):,}",
        f"- Failed drives: {int(drives['failed'].sum()):,}",
        f"- Drive models: {len(models):,}",
        f"- Overall AFR: {afr(drives['failed'].sum(), drives['drive_days'].sum()):.2f}%",
        "- Split by time: train = 2023 Q4 to 2025 Q3, validation = 2025 Q4 to "
        "2026 Q1, test = 2026 Q2 to Q3.",
        "",
        "AFR (annualized failure rate) = failures / (drive-days / 365) x 100%.",
        "",
        "## 1. How many failures does each drive model have?",
        "",
        "Models grouped by their failures across all 12 quarters:",
        "",
        md(tables["buckets_all"]),
        "",
        "Models grouped by their failures in the **test** period only. This is "
        "what limits per-model reporting:",
        "",
        md(tables["buckets_test"]),
        "",
        "## 2. If we keep only models above a cut-off, what do we keep?",
        "",
        md(tables["coverage"]),
        "",
        "## 3. Are failures concentrated in a few models?",
        "",
        md(tables["concentration"]),
        "",
        "## 4. Top 25 drive models by failures",
        "",
        md(top[["brand", "kind", "capacity_tb", "drives", "failures", "afr_pct",
                "failures_train", "failures_valid", "failures_test",
                "quarters_present", "first_quarter", "last_quarter"]]),
        "",
        "## 5. By brand",
        "",
        md(tables["brands"]),
        "",
        "## 6. By capacity (data HDDs)",
        "",
        md(tables["capacity"]),
        "",
        "## 7. By drive age (data HDDs, SMART 9 power-on hours)",
        "",
        "Approximate: each drive-quarter is placed in one age band.",
        "",
        md(tables["age"]),
        "",
        "## 8. Fleet over time: active drives per quarter, top 25 models by drive-days",
        "",
        md(top_fleet),
        "",
        "## 9. Would a shorter training window lose drive models?",
        "",
        "Each row is a training window ending at 2025 Q3. The test period is "
        "fixed at 2026 Q2 to Q3.",
        "",
        md(tables["window"]),
        "",
        "## 10. Model names with no brand rule",
        "",
        (", ".join(f"`{m}`" for m in unmatched) if unmatched else "None."),
        "",
        "Full per-model table: `reports/profile_models.csv`.",
        "",
    ]
    (config.REPORTS_DIR / "failure_profile.md").write_text("\n".join(out))
    print(f"wrote {config.REPORTS_DIR / 'failure_profile.md'}")
