"""Confirm the pipeline works on real data in about 10 minutes.

Downloads one quarter (2023 Q4, about 1 GB), converts it to the declared
schema, validates it, checks its checksum against the frozen benchmark, and
checks its counts against numbers stored here. Then runs the unit tests.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pyarrow.parquet as pq

from . import config
from .benchmark import BENCH_FILE
from .prepare import prepare
from .profile import aggregate_quarter, drives_from
from .schema import validate_file

QUARTER = (2023, 4)
EXPECTED = {"rows": 24_790_999, "drives": 281_346, "failed_drives": 1_057,
            "capacity_not_reported_rows": 8_390}


def main() -> bool:
    name = config.quarter_name(*QUARTER)
    print(f"1/5 download and convert {name} (skipped if already prepared)")
    manifest = prepare([QUARTER])
    path = config.PARQUET_DIR / f"{name}.parquet"

    print("2/5 validate against schema drive_stats_v1")
    report = validate_file(path)
    ok = report.ok
    print("    ok" if report.ok else f"    FAILED: {report.problems}")

    print("3/5 compare the source checksum with benchmark/v1")
    frozen = json.loads(BENCH_FILE.read_text())["data"]["quarters"][name]["zip_sha256"]
    same = manifest[name]["zip_sha256"] == frozen
    ok &= same
    print("    ok" if same else "    DIFFERENT: Backblaze may have republished this file")

    print("4/5 count rows, drives and failures")
    cols = ["date", "serial_number", "model", "capacity_bytes", "failure", "smart_9_raw"]
    t = pq.read_table(path, columns=cols)
    d = drives_from(aggregate_quarter(t, name))
    got = {"rows": t.num_rows, "drives": len(d), "failed_drives": int(d["failed"].sum()),
           "capacity_not_reported_rows": t["capacity_bytes"].null_count}
    for k, v in EXPECTED.items():
        match = got[k] == v
        ok &= match
        print(f"    {k}: {got[k]:,} (expected {v:,}) {'ok' if match else 'MISMATCH'}")

    print("5/5 unit tests")
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=config.ROOT)
    ok &= tests.returncode == 0

    print("\nQUICK CHECK PASSED" if ok else "\nQUICK CHECK FAILED")
    return ok
