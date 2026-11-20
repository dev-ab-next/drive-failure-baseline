"""Download and convert every quarter in the data window.

The next quarter downloads while the current one converts. Zips are
deleted after conversion unless keep_zips is set.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from . import config
from .convert import convert_zip
from .download import download, load_manifest, save_manifest
from .schema import SchemaError, load_declared, validate_file


def prepare(quarters=config.QUARTERS, keep_zips: bool = False) -> dict:
    manifest = load_manifest()
    todo = []
    for year, q in quarters:
        name = config.quarter_name(year, q)
        parquet = config.PARQUET_DIR / f"{name}.parquet"
        if parquet.exists() and name in manifest:
            print(f"{name}: already prepared, skipping")
            continue
        todo.append((year, q, name, parquet))

    def fetch(item):
        year, q, name, _ = item
        zip_path = config.RAW_DIR / f"data_{name}.zip"
        print(f"{name}: downloading")
        info = download(config.quarter_url(year, q), zip_path)
        return zip_path, info

    with ThreadPoolExecutor(max_workers=1) as pool:
        futures = [pool.submit(fetch, todo[0])] if todo else []
        for i, item in enumerate(todo):
            _, _, name, parquet = item
            zip_path, info = futures[i].result()
            if i + 1 < len(todo):
                futures.append(pool.submit(fetch, todo[i + 1]))
            print(f"{name}: converting ({info['bytes'] / 1e9:.2f} GB zip)")
            stats = convert_zip(zip_path, parquet)
            report = validate_file(parquet)
            if not report.ok:
                parquet.rename(parquet.with_suffix(".invalid.parquet"))
                raise SchemaError(f"{name} failed validation: {report.problems}")
            manifest[name] = {
                "url": config.quarter_url(*item[:2]),
                "zip_bytes": info["bytes"],
                "zip_sha256": info["sha256"],
                "rows": stats["rows"],
                "days": stats["days"],
                "columns": stats["columns"],
                "parquet_bytes": parquet.stat().st_size,
                "schema": load_declared().name,
            }
            save_manifest(manifest)
            if not keep_zips:
                zip_path.unlink()
            print(f"{name}: {stats['rows']:,} rows, {stats['days']} days, "
                  f"{len(stats['columns'])} columns")
    schema_report(manifest)
    return manifest


def schema_report(manifest: dict | None = None) -> str:
    """Columns added or dropped between consecutive quarters."""
    manifest = manifest if manifest is not None else load_manifest()
    names = [config.quarter_name(y, q) for y, q in config.QUARTERS
             if config.quarter_name(y, q) in manifest]
    lines = ["# Schema changes between quarters", "",
             "Generated from `data/manifest.json` by `dfb prepare`.", "",
             "| quarter | columns | added | dropped |", "|---|---|---|---|"]
    prev = None
    for n in names:
        cols = manifest[n]["columns"]
        if prev is None:
            added, dropped = "(first quarter)", ""
        else:
            a = [c for c in cols if c not in prev]
            d = [c for c in prev if c not in cols]
            added = ", ".join(f"`{c}`" for c in a) or "none"
            dropped = ", ".join(f"`{c}`" for c in d) or "none"
        lines.append(f"| {n} | {len(cols)} | {added} | {dropped} |")
        prev = cols
    text = "\n".join(lines) + "\n"
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "schema_changes.md").write_text(text)
    return text


def validate_all() -> bool:
    manifest = load_manifest()
    declared = load_declared()
    ok = True
    for year, q in config.QUARTERS:
        name = config.quarter_name(year, q)
        path = config.PARQUET_DIR / f"{name}.parquet"
        if not path.exists():
            print(f"{name}: missing")
            ok = False
            continue
        report = validate_file(path, declared)
        status = "ok" if report.ok else "FAILED: " + "; ".join(report.problems)
        print(f"{name}: {report.rows:,} rows, {report.unknown_capacity:,} with capacity "
              f"not reported, {status}")
        ok &= report.ok
        if report.ok and name in manifest:
            manifest[name]["schema"] = declared.name
            manifest[name]["unknown_capacity_rows"] = report.unknown_capacity
    save_manifest(manifest)
    return ok
