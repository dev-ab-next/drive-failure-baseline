"""The frozen benchmark definition and scoring of outside predictions.

`dfb freeze`   writes benchmark/v1/benchmark.json from the prepared data
`dfb verify`   checks local data against the frozen definition
`dfb test-rows OUT.parquet`  lists the (serial_number, date) rows to score
`dfb evaluate --predictions P.parquet [--valid-predictions V.parquet]`
               scores any method with the benchmark's metric code
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd

from . import config, evaluate
from .features import (CORE, DELTA, DELTA_DAYS, EXTRA, FEATURES_DIR, MAX_WINDOW, SAMPLE_RATE,
                       SAMPLE_SALT)
from .labels import WINDOWS, usable_sql
from .schema import SCHEMA_FILE, load_declared

BENCH_DIR = config.ROOT / "benchmark" / "v1"
BENCH_FILE = BENCH_DIR / "benchmark.json"
VERSION = "benchmark_v1"


def _con():
    c = duckdb.connect()
    try:
        c.execute("SET enable_progress_bar=false")
    except duckdb.InvalidInputException:  # inside Jupyter without ipywidgets
        pass
    return c


def split_counts() -> dict:
    con = _con()
    out = {}
    for period in ("train", "valid", "test"):
        g = f"read_parquet('{FEATURES_DIR}/{period}/*/*.parquet')"
        for n in WINDOWS:
            r = con.execute(f"""SELECT count(*), count(DISTINCT serial_number),
                count(DISTINCT serial_number) FILTER (WHERE days_to_failure = 0),
                count(*) FILTER (WHERE days_to_failure BETWEEN 0 AND {n - 1})
                FROM {g} WHERE {usable_sql(n)}""").fetchone()
            out[f"{period}_N{n}"] = {"rows": r[0], "drives": r[1], "failed_drives": r[2],
                                     "positive_rows": r[3]}
    return out


def definition() -> dict:
    manifest = json.loads(config.MANIFEST.read_text())
    sources = {q: {"url": v["url"], "zip_sha256": v["zip_sha256"], "zip_bytes": v["zip_bytes"],
                   "rows": v["rows"]} for q, v in sorted(manifest.items())}
    return {
        "version": VERSION,
        "data": {"source": "Backblaze Drive Stats (https://www.backblaze.com/cloud-storage/"
                           "resources/hard-drive-test-data); cite Backblaze, do not resell",
                 "quarters": sources,
                 "schema": load_declared().name,
                 "schema_sha256": hashlib.sha256(SCHEMA_FILE.read_bytes()).hexdigest()},
        "scope": "data hard drives only: not SSDs, not drives under 2 TB (decisions R-1, F-7)",
        "splits": {p: {"start": str(a), "end": str(b)} for p, (a, b) in config.PERIODS.items()},
        "split_rule": "by time; a drive may appear in several periods; a row is used only if "
                      "its N-day label is decided inside its own period (S-4, L-2)",
        "label": "positive if the drive fails within the next N days, counting the failure "
                 "day; rows after the failure removed (L-1)",
        "warning_windows_days": list(WINDOWS),
        "main_window_days": 7,
        "features_used_by_baselines": {"core_smart_raw": CORE, "change_of": DELTA,
                                       "change_windows_days": DELTA_DAYS,
                                       "brand_specific_smart_raw": EXTRA},
        "training_sample": {"rate": SAMPLE_RATE, "salt": SAMPLE_SALT,
                            "always_kept": f"days within {MAX_WINDOW} days of a failure"},
        "evaluation": {
            "unit": "drive (B-1, E-1)",
            "false_alarm_levels": list(evaluate.FAR_LEVELS),
            "headline_threshold": "set on validation, applied unchanged to test (E-2)",
            "main_metrics": ["detection rate at fixed false-alarm rate", "days of warning"],
            "other_metrics": ["precision-recall AUC (per drive)", "Brier after isotonic "
                              "calibration on validation", "ROC AUC (legacy only)"],
            "slice_min_failures": evaluate.MIN_SLICE_FAILURES,
        },
        "counts": split_counts(),
    }


def freeze() -> Path:
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    BENCH_FILE.write_text(json.dumps(definition(), indent=2) + "\n")
    return BENCH_FILE


def verify() -> list[str]:
    frozen = json.loads(BENCH_FILE.read_text())
    problems = []
    manifest = json.loads(config.MANIFEST.read_text())
    for q, v in frozen["data"]["quarters"].items():
        if q not in manifest:
            problems.append(f"{q}: not prepared")
        elif manifest[q]["zip_sha256"] != v["zip_sha256"]:
            problems.append(f"{q}: source file differs from the frozen benchmark "
                            "(Backblaze may have republished it)")
    if hashlib.sha256(SCHEMA_FILE.read_bytes()).hexdigest() != frozen["data"]["schema_sha256"]:
        problems.append("schema file differs from the frozen benchmark")
    if (FEATURES_DIR / "test").exists():
        now = split_counts()
        for k, v in frozen["counts"].items():
            if now.get(k) != v:
                problems.append(f"counts differ for {k}: frozen {v}, local {now.get(k)}")
    return problems


def test_rows(out: Path, period: str = "test", n: int = 7) -> int:
    g = f"read_parquet('{FEATURES_DIR}/{period}/*/*.parquet')"
    con = _con()
    con.execute(f"COPY (SELECT serial_number, date FROM {g} WHERE {usable_sql(n)}) "
                f"TO '{out}' (FORMAT parquet)")
    return con.execute(f"SELECT count(*) FROM read_parquet('{out}')").fetchone()[0]


def _scores_for(period: str, predictions: Path, n: int) -> evaluate.DriveScores:
    con = _con()
    parts_d, parts_w, missing = [], [], 0
    for b in range(8):
        df = con.execute(f"""
            SELECT f.serial_number, f.date, f.days_to_failure, f.days_to_period_end,
                   f.brand, f.model, f.capacity_tb, f.age_years, p.score
            FROM read_parquet('{FEATURES_DIR}/{period}/*/part-{b}.parquet') f
            LEFT JOIN read_parquet('{predictions}') p
              ON p.serial_number = f.serial_number AND p.date = f.date
            WHERE {usable_sql(n)}""").df()
        missing += int(df["score"].isna().sum())
        ds = evaluate.drive_scores(df, n, attrs=["brand", "model", "capacity_tb", "age_years"])
        parts_d.append(ds.drives)
        parts_w.append(ds.window)
    if missing:
        raise SystemExit(f"{missing:,} {period} rows have no score; every row from "
                         f"`dfb test-rows` needs one")
    return evaluate.DriveScores(pd.concat(parts_d, ignore_index=True),
                                pd.concat(parts_w, ignore_index=True))


def evaluate_predictions(predictions: Path, valid_predictions: Path | None = None,
                         n: int = 7) -> dict:
    test = _scores_for("test", predictions, n)
    valid = _scores_for("valid", valid_predictions, n) if valid_predictions else None
    out = evaluate.summary(test, valid)
    out["benchmark"] = VERSION
    out["n"] = n
    return out
