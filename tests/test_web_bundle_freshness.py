from __future__ import annotations

import os
from pathlib import Path

from flightstack.web.server import _web_bundle_is_stale


def _write_with_mtime(path: Path, content: str, mtime_ns: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    os.utime(path, ns=(mtime_ns, mtime_ns))


def test_web_bundle_is_stale_when_source_is_newer(tmp_path: Path) -> None:
    web_root = tmp_path / "web" / "dist"
    entry = web_root / "index.html"
    source = tmp_path / "web" / "src" / "main.ts"

    _write_with_mtime(entry, "<title>old build</title>", 1_000_000_000)
    _write_with_mtime(source, "console.log('new source')", 2_000_000_000)

    assert _web_bundle_is_stale(web_root)


def test_web_bundle_is_fresh_after_rebuild(tmp_path: Path) -> None:
    web_root = tmp_path / "web" / "dist"
    entry = web_root / "index.html"
    source = tmp_path / "web" / "src" / "main.ts"

    _write_with_mtime(source, "console.log('source')", 1_000_000_000)
    _write_with_mtime(entry, "<title>new build</title>", 2_000_000_000)

    assert not _web_bundle_is_stale(web_root)
