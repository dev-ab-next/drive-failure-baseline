"""Declared Parquet schema and validation (decision D-6).

Every per-quarter Parquet file must have exactly the columns, order and
types in `schema/drive_stats_v1.json`. Files are checked when written
(`dfb prepare`) and before being read (`dfb profile`, later steps), so a
schema drift in a new Backblaze release stops the pipeline instead of
silently changing results.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import config

SCHEMA_FILE = config.ROOT / "schema" / "drive_stats_v1.json"
METADATA_KEY = b"dfb_schema"

TYPES = {"date32": pa.date32(), "string": pa.string(), "int64": pa.int64(),
         "int8": pa.int8(), "float64": pa.float64()}


class SchemaError(ValueError):
    """A file or CSV does not match the declared schema."""


@dataclass(frozen=True)
class DeclaredSchema:
    name: str
    arrow: pa.Schema
    required: tuple[str, ...]  # non-nullable columns

    @property
    def names(self) -> list[str]:
        return self.arrow.names


def load_declared(path: Path = SCHEMA_FILE) -> DeclaredSchema:
    spec = json.loads(path.read_text())
    fields = [pa.field(c["name"], TYPES[c["type"]], nullable=c["nullable"])
              for c in spec["columns"]]
    schema = pa.schema(fields, metadata={METADATA_KEY: spec["name"].encode()})
    required = tuple(c["name"] for c in spec["columns"] if not c["nullable"])
    return DeclaredSchema(spec["name"], schema, required)


def check_columns(columns: list[str], declared: DeclaredSchema, where: str) -> None:
    """Fail if a source has a column the declared schema does not know."""
    unknown = [c for c in columns if c and c not in declared.names]
    if unknown:
        raise SchemaError(
            f"{where}: columns not in schema {declared.name}: {unknown}. "
            "Add them to schema/drive_stats_v1.json (as a new schema version) "
            "and record the change in docs/decisions.md before converting.")


# Backblaze writes -1 when a drive did not report its capacity that day
# (decision D-8). It is stored as empty; any other value <= 0 fails validation.
UNKNOWN_CAPACITY = -1


def conform(table: pa.Table, declared: DeclaredSchema) -> pa.Table:
    """Return `table` with exactly the declared columns, order and types.

    Missing nullable columns are filled with nulls; a missing required
    column, an unknown column or a failed cast raises SchemaError.
    `capacity_bytes == -1` becomes null (decision D-8).
    """
    check_columns(table.column_names, declared, "table")
    arrays = []
    for f in declared.arrow:
        if f.name in table.column_names:
            try:
                col = table[f.name].cast(f.type)
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
                raise SchemaError(f"column {f.name}: cannot cast to {f.type}: {exc}") from exc
            if f.name == "capacity_bytes":
                unknown = pc.equal(col, UNKNOWN_CAPACITY)
                col = pc.if_else(unknown, pa.scalar(None, f.type), col)
            arrays.append(col)
        elif f.nullable:
            arrays.append(pa.nulls(table.num_rows, type=f.type))
        else:
            raise SchemaError(f"required column {f.name} is missing")
    return pa.Table.from_arrays(arrays, schema=declared.arrow)


def check_file_schema(path: Path, declared: DeclaredSchema | None = None) -> None:
    """Fast check (reads only the footer): names, order, types, version tag."""
    declared = declared or load_declared()
    actual = pq.read_schema(path)
    tag = (actual.metadata or {}).get(METADATA_KEY)
    if tag != declared.name.encode():
        raise SchemaError(f"{path.name}: schema tag is {tag!r}, expected {declared.name!r}; "
                          "re-run `dfb prepare`")
    if actual.names != declared.names:
        missing = [c for c in declared.names if c not in actual.names]
        extra = [c for c in actual.names if c not in declared.names]
        raise SchemaError(f"{path.name}: columns differ (missing {missing}, extra {extra}, "
                          "or order changed)")
    for a, d in zip(actual, declared.arrow):
        if a.type != d.type:
            raise SchemaError(f"{path.name}: column {a.name} is {a.type}, expected {d.type}")


@dataclass
class ValidationReport:
    path: str
    rows: int = 0
    unknown_capacity: int = 0
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def quarter_bounds(name: str) -> tuple[dt.date, dt.date]:
    q, year = int(name[1]), int(name[3:])
    start = dt.date(year, 3 * (q - 1) + 1, 1)
    end = (dt.date(year + 1, 1, 1) if q == 4 else dt.date(year, 3 * q + 1, 1)) - dt.timedelta(days=1)
    return start, end


def validate_file(path: Path, declared: DeclaredSchema | None = None) -> ValidationReport:
    """Full check: schema, plus value rules on the required columns.

    Rules: no nulls in required columns; `failure` is 0 or 1;
    `capacity_bytes` is positive where present (empty = not reported,
    decision D-8); every date lies in the file's quarter; no
    (date, serial_number) pair appears twice.
    """
    declared = declared or load_declared()
    report = ValidationReport(str(path))
    try:
        check_file_schema(path, declared)
    except SchemaError as exc:
        report.problems.append(str(exc))
        return report
    t = pq.read_table(path, columns=list(dict.fromkeys([*declared.required, "capacity_bytes"])))
    report.rows = t.num_rows
    for c in declared.required:
        if t[c].null_count:
            report.problems.append(f"{c}: {t[c].null_count:,} nulls in a required column")
    bad_fail = pc.sum(pc.invert(pc.is_in(t["failure"], pa.array([0, 1], pa.int8())))).as_py()
    if bad_fail:
        report.problems.append(f"failure: {bad_fail:,} values other than 0 or 1")
    bad_cap = pc.sum(pc.less_equal(t["capacity_bytes"], 0)).as_py() or 0
    if bad_cap:
        report.problems.append(f"capacity_bytes: {bad_cap:,} values <= 0")
    report.unknown_capacity = t["capacity_bytes"].null_count
    start, end = quarter_bounds(path.stem)
    lo, hi = pc.min_max(t["date"]).values()
    if lo.as_py() < start or hi.as_py() > end:
        report.problems.append(f"date: range {lo} to {hi} falls outside {start} to {end}")
    dup = t.select(["date", "serial_number"]).group_by(["date", "serial_number"]).aggregate(
        [([], "count_all")])
    n_dup = pc.sum(pc.greater(dup["count_all"], 1)).as_py() or 0
    if n_dup:
        report.problems.append(f"(date, serial_number): {n_dup:,} pairs appear more than once")
    return report
