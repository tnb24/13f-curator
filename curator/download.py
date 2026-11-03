"""Filing download: store each filing's XML documents under output/filings.

Every filing keeps exactly one file at ``{accession}.xml``. That file holds
the information table when the filing has one, and the cover page when the
filing is a notice. Regular filings also keep the cover page at
``{accession}.primary_doc.xml``. Bytes are written exactly as received.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

from lxml import etree
from lxml.etree import QName

from curator.cache import Cache
from curator.records import FilingDocs, FilingRef

_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/"
_INDEX_NAME = "index.json"
_COVER_FILENAME = "primary_doc.xml"
_INFO_TABLE_LOCALNAME = "informationTable"
_COVER_LOCALNAME = "edgarSubmission"


def download(refs: list[FilingRef], cache: Cache, out_dir: Path) -> list[FilingDocs]:
    """Download every ref once and store its documents. Return docs in ref order."""
    out_dir = Path(out_dir)
    return [_download_one(ref, cache, out_dir) for ref in refs]


def _download_one(ref: FilingRef, cache: Cache, out_dir: Path) -> FilingDocs:
    dir_url = _ARCHIVE_URL.format(cik=int(ref.cik), accession=ref.accession.replace("-", ""))
    bodies, localnames = _read_documents(cache, dir_url)

    info_name = next(
        (name for name, tag in localnames.items() if tag == _INFO_TABLE_LOCALNAME), None
    )
    fallback_cover_name = next(
        (name for name, tag in localnames.items() if tag == _COVER_LOCALNAME), None
    )
    if _COVER_FILENAME in bodies:
        cover_name: str | None = _COVER_FILENAME
    else:
        cover_name = fallback_cover_name

    if cover_name is None:
        raise RuntimeError(
            f"{ref.accession}: no cover page in {dir_url} "
            f"(no {_COVER_FILENAME}, no {_COVER_LOCALNAME} root)"
        )

    target_dir = out_dir / str(int(ref.cik))
    stem = ref.accession.replace("-", "")
    main_path = target_dir / f"{stem}.xml"
    primary_path = target_dir / f"{stem}.primary_doc.xml"

    if info_name is None:
        # A notice has no information table. {accession}.xml holds its cover page.
        _write_bytes(main_path, bodies[cover_name])
        print(
            f"curator.download: {ref.cik} {ref.accession} {ref.form}: "
            f"notice cover page -> {main_path.name}",
            file=sys.stderr,
        )
        return FilingDocs(ref=ref, info_table_path=None, primary_doc_path=main_path)

    _write_bytes(main_path, bodies[info_name])
    _write_bytes(primary_path, bodies[cover_name])
    print(
        f"curator.download: {ref.cik} {ref.accession} {ref.form}: "
        f"information table -> {main_path.name}, cover page -> {primary_path.name}",
        file=sys.stderr,
    )
    return FilingDocs(ref=ref, info_table_path=main_path, primary_doc_path=primary_path)


def _read_documents(cache: Cache, dir_url: str) -> tuple[dict[str, bytes], dict[str, str | None]]:
    """Fetch index.json and every .xml document. Return bodies and root local names."""
    index = json.loads(cache.get(dir_url + _INDEX_NAME))
    items = (index.get("directory") or {}).get("item") or []
    names = [
        str(item.get("name"))
        for item in items
        if isinstance(item, dict)
        and str(item.get("name", "")).lower().endswith(".xml")
    ]
    bodies: dict[str, bytes] = {}
    localnames: dict[str, str | None] = {}
    for name in names:
        body = cache.get(dir_url + name)
        bodies[name] = body
        localnames[name] = root_localname(body)
    return bodies, localnames


def root_localname(body: bytes) -> str | None:
    """Return the local name of the root element, or None when no root parses.

    The parser stops at the first start tag. An information table can be 30 MB,
    and a full parse here only to read one tag name doubles the parse work.
    """
    try:
        for _event, element in etree.iterparse(
            io.BytesIO(body), events=("start",), resolve_entities=False, no_network=True
        ):
            return QName(element).localname
    except etree.XMLSyntaxError:
        # A directory can hold XML that is not a 13F document. Skip it.
        return None
    return None


def _write_bytes(path: Path, body: bytes) -> None:
    """Write body in binary mode. Skip the write when the file already holds it."""
    if path.is_file() and path.read_bytes() == body:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
