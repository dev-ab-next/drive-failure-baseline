"""Convert a quarter zip of daily CSVs into one Parquet file.

Every file is written in the declared schema (`schema/drive_stats_v1.json`,
decision D-6): same columns, order and types for every quarter. A CSV
column the schema does not know stops conversion. Days are read one at a
time to keep memory small.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path, PurePosixPath

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from .schema import DeclaredSchema, check_columns, conform, load_declared


def daily_members(zf: zipfile.ZipFile) -> list[str]:
    """CSV files in the zip, skipping macOS metadata, sorted by name (= date)."""
    names = []
    for name in zf.namelist():
        p = PurePosixPath(name)
        if p.suffix.lower() != ".csv":
            continue
        if "__MACOSX" in p.parts or p.name.startswith("._"):
            continue
        names.append(name)
    return sorted(names, key=lambda n: PurePosixPath(n).name)


def read_header(zf: zipfile.ZipFile, member: str) -> list[str]:
    with zf.open(member) as fh:
        line = fh.readline().decode("utf-8-sig").strip()
    return [c.strip() for c in line.split(",")]


def read_day(zf: zipfile.ZipFile, member: str, declared: DeclaredSchema) -> pa.Table:
    header = read_header(zf, member)
    check_columns(header, declared, member)
    types = {f.name: f.type for f in declared.arrow if f.name in header}
    with zf.open(member) as fh:
        table = pacsv.read_csv(
            io.BytesIO(fh.read()),
            convert_options=pacsv.ConvertOptions(column_types=types),
        )
    return conform(table, declared)


def convert_zip(zip_path: Path, out_path: Path,
                declared: DeclaredSchema | None = None) -> dict:
    """Write `out_path` (Parquet) in the declared schema. Returns stats."""
    declared = declared or load_declared()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".parquet.part")
    rows = 0
    source_columns: list[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        members = daily_members(zf)
        if not members:
            raise ValueError(f"no CSV files in {zip_path}")
        for m in members:  # fail fast, before writing anything
            header = read_header(zf, m)
            check_columns(header, declared, m)
            source_columns += [c for c in header if c and c not in source_columns]
        with pq.ParquetWriter(tmp, declared.arrow, compression="zstd") as writer:
            for member in members:
                table = read_day(zf, member, declared)
                writer.write_table(table, row_group_size=len(table) or None)
                rows += table.num_rows
    tmp.replace(out_path)
    return {"rows": rows, "days": len(members), "columns": source_columns}

