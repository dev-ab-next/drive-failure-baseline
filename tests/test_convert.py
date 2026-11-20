import zipfile

import pyarrow.parquet as pq

from dfb.convert import convert_zip


def test_schema_union_fills_missing_columns(tmp_path):
    z = tmp_path / "q.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("data/2024-01-01.csv",
                    "date,serial_number,model,capacity_bytes,failure,smart_5_raw\n"
                    "2024-01-01,A,M1,100,0,3\n")
        zf.writestr("data/2024-01-02.csv",
                    "date,serial_number,model,capacity_bytes,failure,smart_9_raw,datacenter\n"
                    "2024-01-02,A,M1,100,1,500,phx1\n")
        zf.writestr("__MACOSX/data/._2024-01-02.csv", "junk")
    out = tmp_path / "q.parquet"
    stats = convert_zip(z, out)
    t = pq.read_table(out).to_pydict()
    assert stats["days"] == 2 and stats["rows"] == 2
    assert t["smart_5_raw"] == [3.0, None]
    assert t["smart_9_raw"] == [None, 500.0]
    assert t["datacenter"] == [None, "phx1"]
    assert t["failure"] == [0, 1]
