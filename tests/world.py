"""A small synthetic EDGAR, served from memory, for end-to-end pipeline tests.

The world holds three listed filers and one parent manager:

    Alpha Capital LLC     CIK 1000001  input CIK right; 13F-HR in Q1 and Q2
    Beta Partners LP      CIK 1000002  input says 1000099; two lookup entries
                                       share its name, and only 1000002 files
    Gamma Advisors Inc    CIK 1000003  13F-HR in Q1, 13F-NT in Q2
    Delta Holdings LLC    CIK 1000004  not in the list; its Q2 13F-HR holds
                                       Gamma's rows as other-manager sequence 1

`bodies()` returns every URL the pipeline requests, mapped to its body.
`FakeSource` serves them and records each request.
"""

from __future__ import annotations

import json
from pathlib import Path

LOOKUP_URL = "https://www.sec.gov/Archives/edgar/cik-lookup-data.txt"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/"
Q2_LIST_URL = "https://www.sec.gov/files/investment/13flist2026q2-txt.txt"
Q1_LIST_URL = "https://www.sec.gov/files/investment/13flist2026q1.txt"

ROSTER = (
    "fund_name,cik\n"
    "Alpha Capital LLC,1000001\n"
    "Beta Partners LP,1000099\n"
    "Gamma Advisors Inc,1000003\n"
)

APPLE = ("APPLE INC", "COM", "037833100")
NVIDIA = ("NVIDIA CORPORATION", "COM", "67066G104")
ACME = ("ACME WIDGET CO", "COM", "000360206")
FOREIGN = ("GLOBAL SHIPPING LTD", "SHS", "G3922B107")
TYPO = ("TYPO CORP", "COM", "037833101")  # Apple's CUSIP with a bad check digit
OLDCO = ("OLDCO INC", "COM", "123456782")   # on the Q1 list only


class FakeSource:
    """Serve canned bodies by URL. Record every request. Fail on an unknown URL."""

    def __init__(self, bodies: dict[str, bytes]) -> None:
        self._bodies = bodies
        self.requested: list[str] = []

    def get(self, url: str) -> bytes:
        from curator.cache import FetchError

        self.requested.append(url)
        if url not in self._bodies:
            raise FetchError(f"GET {url} failed (no canned body)")
        return self._bodies[url]


def write_roster(directory: Path) -> Path:
    path = Path(directory) / "filers.csv"
    path.write_text(ROSTER, encoding="utf-8")
    return path


def _cover(cik: int, name: str, period: str, report_type: str,
           entries: int | None, total: int | None, others: str = "", parents: str = "") -> bytes:
    summary = "" if entries is None else f"""
    <summaryPage>
      <otherIncludedManagersCount>0</otherIncludedManagersCount>
      <tableEntryTotal>{entries}</tableEntryTotal>
      <tableValueTotal>{total}</tableValueTotal>
      {others}
    </summaryPage>"""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<edgarSubmission xmlns="http://www.sec.gov/edgar/thirteenffiler">
  <headerData><filerInfo><filer><credentials>
    <cik>{cik:010d}</cik><ccc>XXXXXXXX</ccc>
  </credentials></filer></filerInfo></headerData>
  <formData>
    <coverPage>
      <reportCalendarOrQuarter>{period}</reportCalendarOrQuarter>
      <filingManager><name>{name}</name></filingManager>
      <reportType>{report_type}</reportType>
      <form13FFileNumber>028-{cik % 100000:05d}</form13FFileNumber>
      {parents}
    </coverPage>{summary}
  </formData>
</edgarSubmission>
""".encode()


def _row(issuer: tuple[str, str, str], value: int, shares: int,
         put_call: str | None = None, other: str | None = None) -> str:
    name, title, cusip = issuer
    option = f"<putCall>{put_call}</putCall>" if put_call else ""
    manager = f"<otherManager>{other}</otherManager>" if other else ""
    return f"""
  <infoTable>
    <nameOfIssuer>{name}</nameOfIssuer><titleOfClass>{title}</titleOfClass>
    <cusip>{cusip}</cusip><value>{value}</value>
    <shrsOrPrnAmt><sshPrnamt>{shares}</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt>
    {option}<investmentDiscretion>SOLE</investmentDiscretion>{manager}
    <votingAuthority><Sole>{shares}</Sole><Shared>0</Shared><None>0</None></votingAuthority>
  </infoTable>"""


def _table(*rows: str) -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<ns1:informationTable xmlns:ns1="http://www.sec.gov/edgar/document/thirteenf/informationtable">'
        + "".join(rows).replace("<", "<ns1:").replace("<ns1:/", "</ns1:")
        + "\n</ns1:informationTable>\n"
    ).replace("<ns1:?xml", "<?xml").encode()


# (cik, name, accession, form, report date, filing date, cover, table or None)
_FILINGS = [
    (1000001, "ALPHA CAPITAL LLC", "0001000001-26-000001", "13F-HR", "2026-03-31", "2026-05-10",
     ("03-31-2026", "13F HOLDINGS REPORT", 2, 3000),
     [_row(APPLE, 1000, 10), _row(NVIDIA, 2000, 20)]),
    (1000001, "ALPHA CAPITAL LLC", "0001000001-26-000002", "13F-HR", "2026-06-30", "2026-08-10",
     ("06-30-2026", "13F HOLDINGS REPORT", 3, 6500),
     [_row(APPLE, 1500, 12), _row(APPLE, 4000, 40, put_call="Call"), _row(ACME, 1000, 5)]),
    (1000002, "BETA PARTNERS LP", "0001000002-26-000001", "13F-HR", "2026-03-31", "2026-05-12",
     ("03-31-2026", "13F HOLDINGS REPORT", 1, 700),
     [_row(NVIDIA, 700, 7)]),
    (1000002, "BETA PARTNERS LP", "0001000002-26-000002", "13F-HR", "2026-06-30", "2026-08-12",
     ("06-30-2026", "13F HOLDINGS REPORT", 4, 1030),
     [_row(NVIDIA, 900, 9), _row(FOREIGN, 50, 1), _row(TYPO, 30, 3), _row(OLDCO, 50, 5)]),
    (1000003, "GAMMA ADVISORS INC", "0001000003-26-000001", "13F-HR", "2026-03-31", "2026-05-14",
     ("03-31-2026", "13F HOLDINGS REPORT", 1, 300),
     [_row(ACME, 300, 3)]),
    (1000003, "GAMMA ADVISORS INC", "0001000003-26-000002", "13F-NT", "2026-06-30", "2026-08-14",
     ("06-30-2026", "13F NOTICE", None, None), None),
    (1000004, "DELTA HOLDINGS LLC", "0001000004-26-000001", "13F-HR", "2026-06-30", "2026-08-13",
     ("06-30-2026", "13F COMBINATION REPORT", 3, 3600),
     [_row(APPLE, 1000, 10, other="1"), _row(NVIDIA, 2000, 20, other="11"),
      _row(ACME, 600, 6, other="2, 1")]),
]

_GAMMA_PARENT = """<otherManagersInfo><otherManager>
        <cik>0001000004</cik><form13FFileNumber>028-00004</form13FFileNumber>
        <name>DELTA HOLDINGS LLC</name></otherManager></otherManagersInfo>"""
_DELTA_OTHERS = """<otherManagers2Info>
        <otherManager2><sequenceNumber>1</sequenceNumber><otherManager>
          <cik>0001000003</cik><form13FFileNumber>028-00003</form13FFileNumber>
          <name>GAMMA ADVISORS INC</name></otherManager></otherManager2>
        <otherManager2><sequenceNumber>2</sequenceNumber><otherManager>
          <cik>0001999999</cik><form13FFileNumber>028-99999</form13FFileNumber>
          <name>SOMEONE ELSE</name></otherManager></otherManager2>
      </otherManagers2Info>"""


def _official(entries: list[tuple[str, str, str]], added: str = "") -> bytes:
    lines = []
    for cusip, issuer, title in entries:
        flag = " *A*" if cusip == added else ""
        lines.append(f"{cusip:<9} {issuer:<30}{(title + flag):<30}")
    return ("\n".join(lines) + "\n").encode("latin-1")


def bodies() -> dict[str, bytes]:
    out: dict[str, bytes] = {
        LOOKUP_URL: (
            "ALPHA CAPITAL LLC:1000001:\n"
            "BETA PARTNERS LP:1000002:\n"
            "BETA PARTNERS LP /ADV:1000098:\n"
            "GAMMA ADVISORS INC:1000003:\n"
            "DELTA HOLDINGS LLC:1000004:\n"
            "UNRELATED & SONS, INC.:1234567:\n"
        ).encode("latin-1"),
        SUBMISSIONS_URL.format(1000098): json.dumps({
            "name": "BETA PARTNERS LP /ADV",
            "filings": {"recent": {"accessionNumber": ["0001000098-26-000001"],
                                   "form": ["ADV"], "reportDate": [""],
                                   "filingDate": ["2026-02-01"], "primaryDocument": ["a.htm"]}},
        }).encode(),
        Q2_LIST_URL: _official([
            (APPLE[2], "APPLE INC", "COM"),
            (NVIDIA[2], "NVIDIA CORPORATION", "COM"),
            (ACME[2], "ACME WIDGET CO", "COM"),
        ], added=ACME[2]),
        Q1_LIST_URL: _official([
            (APPLE[2], "APPLE INC", "COM"),
            (NVIDIA[2], "NVIDIA CORP", "COM"),
            (OLDCO[2], "OLDCO INC", "COM"),
        ]),
    }
    by_cik: dict[int, list] = {}
    for filing in _FILINGS:
        by_cik.setdefault(filing[0], []).append(filing)
    for cik, filings in by_cik.items():
        out[SUBMISSIONS_URL.format(cik)] = json.dumps({
            "name": filings[0][1],
            "filings": {"recent": {
                "accessionNumber": [f[2] for f in filings],
                "form": [f[3] for f in filings],
                "reportDate": [f[4] for f in filings],
                "filingDate": [f[5] for f in filings],
                "primaryDocument": ["primary_doc.xml" for _ in filings],
            }},
        }).encode()
        for _, name, accession, _form, _report, _filed, cover, rows in filings:
            period, report_type, entries, total = cover
            others = _DELTA_OTHERS if cik == 1000004 else ""
            parents = _GAMMA_PARENT if report_type == "13F NOTICE" else ""
            base = ARCHIVE_URL.format(cik=cik, accession=accession.replace("-", ""))
            names = ["primary_doc.xml"] + (["infotable.xml"] if rows else [])
            out[base + "index.json"] = json.dumps(
                {"directory": {"item": [{"name": n} for n in names]}}).encode()
            out[base + "primary_doc.xml"] = _cover(
                cik, name, period, report_type, entries, total, others, parents)
            if rows:
                out[base + "infotable.xml"] = _table(*rows)
    return out
