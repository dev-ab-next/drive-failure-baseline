import datetime as dt
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dfb.convert import convert_zip
from dfb.schema import SchemaError, check_file_schema, conform, load_declared, validate_file

HEADER = "date,serial_number,model,capacity_bytes,failure,smart_9_raw"


def make_zip(path, rows, header=HEADER, name="data/2024-01-01.csv"):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(name, header + "\n" + "\n".join(rows) + "\n")
    return path


def test_declared_schema_loads():
    d = load_declared()
    assert d.name == "drive_stats_v1"
    assert d.names[:5] == ["date", "serial_number", "model", "capacity_bytes", "failure"]
    assert set(d.required) == {"date", "serial_number", "model", "failure"}
    assert all(f.type == pa.float64() for f in d.arrow if f.name.startswith("smart_"))


def test_every_file_has_full_declared_schema(tmp_path):
    z = make_zip(tmp_path / "x.zip", ["2024-01-01,A,M,100,0,5"])
    out = tmp_path / "Q1_2024.parquet"
    convert_zip(z, out)
    check_file_schema(out)  # same names, order, types and tag as declared
    assert validate_file(out).ok


def test_unknown_column_stops_conversion(tmp_path):
    z = make_zip(tmp_path / "x.zip", ["2024-01-01,A,M,100,0,5"],
                 header="date,serial_number,model,capacity_bytes,failure,smart_999_raw")
    with pytest.raises(SchemaError, match="smart_999_raw"):
        convert_zip(z, tmp_path / "Q1_2024.parquet")
    assert not (tmp_path / "Q1_2024.parquet").exists()


def test_missing_required_column_is_rejected():
    t = pa.table({"date": pa.array([dt.date(2024, 1, 1)]), "serial_number": ["A"],
                  "capacity_bytes": [1], "failure": pa.array([0], pa.int8())})
    with pytest.raises(SchemaError, match="model"):
        conform(t, load_declared())


def test_bad_value_cannot_cast(tmp_path):
    z = make_zip(tmp_path / "x.zip", ["2024-01-01,A,M,100,0,notanumber"])
    with pytest.raises((SchemaError, pa.ArrowInvalid)):
        convert_zip(z, tmp_path / "Q1_2024.parquet")


@pytest.mark.parametrize("rows,problem", [
    (["2024-01-01,A,M,100,2,5"], "failure"),
    (["2024-01-01,A,M,0,0,5"], "capacity_bytes"),
    (["2024-01-01,A,M,-5,0,5"], "capacity_bytes"),
    (["2024-05-01,A,M,100,0,5"], "outside"),
    (["2024-01-01,A,M,100,0,5", "2024-01-01,A,M,100,0,6"], "more than once"),
])
def test_value_rules(tmp_path, rows, problem):
    z = make_zip(tmp_path / "x.zip", rows)
    out = tmp_path / "Q1_2024.parquet"
    convert_zip(z, out)
    report = validate_file(out)
    assert not report.ok and any(problem in p for p in report.problems)


def test_file_in_old_schema_is_refused(tmp_path):
    old = pa.table({"date": pa.array([dt.date(2024, 1, 1)]), "serial_number": ["A"],
                    "model": ["M"], "capacity_bytes": pa.array([100], pa.int64()),
                    "failure": pa.array([0], pa.int8())})
    path = tmp_path / "Q1_2024.parquet"
    pq.write_table(old, path)
    with pytest.raises(SchemaError):
        check_file_schema(path)


def test_capacity_minus_one_means_not_reported(tmp_path):
    z = make_zip(tmp_path / "x.zip", ["2024-01-01,A,M,-1,0,5", "2024-01-02,A,M,100,0,5"])
    out = tmp_path / "Q1_2024.parquet"
    convert_zip(z, out)
    assert pq.read_table(out)["capacity_bytes"].to_pylist() == [None, 100]
    report = validate_file(out)
    assert report.ok and report.unknown_capacity == 1
