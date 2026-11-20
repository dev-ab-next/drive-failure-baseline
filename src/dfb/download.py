"""Fetch Backblaze quarter zips from the official source."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from pathlib import Path

from . import config

CHUNK = 8 * 1024 * 1024


def load_manifest(path: Path = config.MANIFEST) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_manifest(manifest: dict, path: Path = config.MANIFEST) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    tmp.replace(path)


def download(url: str, dest: Path, retries: int = 4) -> dict:
    """Stream `url` to `dest`. Returns {'bytes', 'sha256'}. Retries with backoff."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries + 1):
        try:
            digest = hashlib.sha256()
            size = 0
            with urllib.request.urlopen(url, timeout=120) as resp, open(part, "wb") as out:
                while chunk := resp.read(CHUNK):
                    out.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            part.replace(dest)
            return {"bytes": size, "sha256": digest.hexdigest()}
        except Exception as exc:  # network errors: retry with backoff
            if attempt == retries:
                raise
            wait = 2 ** (attempt + 1)
            print(f"  download failed ({exc}); retrying in {wait}s")
            time.sleep(wait)
    raise AssertionError("unreachable")
