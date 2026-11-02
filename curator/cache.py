"""Disk cache for raw EDGAR responses and the per-run manifest.

The cache key is the SHA-256 hex digest of the URL. Each entry holds the
body in ``.bin`` and a sidecar with metadata in ``.json``. A sidecar
always describes a complete body, because the code writes the body first.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from curator.http import Fetcher

_HTTP_SUBDIR = "http"


def _utc_now_z() -> str:
    """Return now in UTC as second precision with a literal Z suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Cache:
    """Return cached bodies. Fetch and store each body on a miss."""

    def __init__(self, root: Path, fetcher: Fetcher, manifest: "Manifest") -> None:
        self._root = Path(root)
        self._dir = self._root / _HTTP_SUBDIR
        self._fetcher = fetcher
        self._manifest = manifest

    @property
    def root(self) -> Path:
        """The cache directory. Other stages keep their derived files here."""
        return self._root

    def get(self, url: str) -> bytes:
        """Return the cached body. Fetch and store it on a miss."""
        key = hashlib.sha256(url.encode("utf-8")).hexdigest()
        bin_path = self._dir / f"{key}.bin"
        meta_path = self._dir / f"{key}.json"
        if bin_path.is_file() and meta_path.is_file():
            body = bin_path.read_bytes()
            self._manifest.record(url, True, len(body), 0.0)
            return body
        started_at = time.perf_counter()
        body = self._fetcher.get(url)
        elapsed_s = time.perf_counter() - started_at
        self._store(key, url, body)
        self._manifest.record(url, False, len(body), elapsed_s)
        return body

    def _store(self, key: str, url: str, body: bytes) -> None:
        """Write the .bin first and the sidecar last through temp files."""
        bin_path = self._dir / f"{key}.bin"
        meta_path = self._dir / f"{key}.json"
        meta = {
            "url": url,
            "status": 200,
            "bytes": len(body),
            "fetched_at": _utc_now_z(),
        }
        self._dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_bytes(bin_path, body)
        _atomic_write_bytes(meta_path, json.dumps(meta, indent=2).encode("utf-8"))


def _atomic_write_bytes(target: Path, payload: bytes) -> None:
    """Write payload to a temp file in the target directory, then rename it."""
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=target.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


_ENTRY_FIELDS = ("url", "hit", "bytes", "elapsed_seconds")
_RUN_FIELDS = (
    "started_at",
    "elapsed_seconds",
    "requests",
    "cache_hits",
    "cache_misses",
    "bytes_downloaded",
)


class Manifest:
    """Record one entry per request and write the fixed-field manifest."""

    def __init__(self) -> None:
        self.started_at = _utc_now_z()
        self._start_monotonic = time.monotonic()
        self._entries: list[dict] = []

    def record(self, url: str, hit: bool, size_bytes: int, elapsed_s: float) -> None:
        """Record one call to Cache.get as a hit or as a miss."""
        self._entries.append(
            {
                "url": url,
                "hit": hit,
                "bytes": size_bytes,
                "elapsed_seconds": round(elapsed_s, 3),
            }
        )

    @property
    def requests(self) -> int:
        return len(self._entries)

    @property
    def cache_hits(self) -> int:
        return sum(1 for entry in self._entries if entry["hit"])

    @property
    def cache_misses(self) -> int:
        return sum(1 for entry in self._entries if not entry["hit"])

    @property
    def bytes_downloaded(self) -> int:
        return sum(entry["bytes"] for entry in self._entries if not entry["hit"])

    def write(self, path: Path) -> None:
        """Sort the entries by URL and write the manifest file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        run = {
            "started_at": self.started_at,
            "elapsed_seconds": round(time.monotonic() - self._start_monotonic, 1),
            "requests": self.requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "bytes_downloaded": self.bytes_downloaded,
        }
        entries = sorted(self._entries, key=lambda entry: entry["url"])
        document = {"run": {field: run[field] for field in _RUN_FIELDS},
                    "entries": [{field: e[field] for field in _ENTRY_FIELDS} for e in entries]}
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
