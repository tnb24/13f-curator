# Output Schema Specification

Your pipeline must produce **two Parquet files**:

| File | Grain | Expected rows |
|---|---|---|
| `filings.parquet` | One row per 13F filing | 40 |
| `holdings.parquet` | One row per position | varies (thousands) |

They are related by `accession_number`.

**Why two files rather than one flat table:** not every filing contains holdings. A `13F-NT` (notice) filing has a cover page but no information table. In a single flat table such a filing would produce zero rows and vanish from your output entirely. Keeping filings separate means every filing is represented regardless of whether it carries positions.

Additional columns beyond those specified are permitted. Required columns must use the exact names and types below.

---

## `filings.parquet`

One row per filing. Primary key: `accession_number`.

| Column | Type | Null? | Description |
|---|---|---|---|
| `accession_number` | `string` | No | EDGAR's unique identifier for the filing, in dashed form (`0001234567-26-000123`). Obtain this from the submissions API — do not construct it. |
| `cik` | `string` | No | Central Index Key of the filing manager, zero-padded to 10 characters. String, not integer: leading zeros are significant. |
| `fund_name` | `string` | No | Fund name as given in the supplied roster CSV. Use this for joining back to the roster; it may differ from `filing_manager`. |
| `filing_manager` | `string` | No | Manager name exactly as it appears on the filing's cover page. May differ from `fund_name` — the legal filing entity is not always the name a fund is known by. |
| `form_type` | `string` | No | The EDGAR form type: `13F-HR`, `13F-HR/A`, `13F-NT`, or `13F-NT/A`. |
| `report_period` | `date32` | No | Last calendar day of the quarter the filing covers (e.g. `2026-03-31`). Parsed from the cover page, which formats it `MM-DD-YYYY`. |
| `report_quarter` | `string` | No | The reporting period as `YYYYQN` (e.g. `2026Q1`). Derived from `report_period`. |
| `filing_date` | `date32` | No | Date the filing was accepted by EDGAR. Distinct from `report_period` — a Q1 filing is submitted up to 45 days after the quarter ends. |
| `is_amendment` | `bool` | No | True when `form_type` ends in `/A`. |
| `amendment_no` | `int32` | Yes | Sequence number of the amendment. Null for original filings. |
| `amendment_type` | `string` | Yes | `RESTATEMENT` (replaces the original filing in full) or `NEW HOLDINGS` (adds to it). Null for original filings. This distinction matters: a restatement supersedes its original. |
| `report_type` | `string` | No | Cover page report type: `13F HOLDINGS REPORT`, `13F NOTICE`, or `13F COMBINATION REPORT`. |
| `form_13f_file_number` | `string` | Yes | The manager's 13F file number (`028-NNNNN`), assigned by the SEC. |
| `crd_number` | `string` | Yes | FINRA Central Registration Depository number. String — leading zeros are significant. |
| `sec_file_number` | `string` | Yes | The manager's SEC file number (`801-NNNNNN`). |
| `other_included_managers_count` | `int32` | Yes | Number of other managers whose holdings are included in this filing. `0` when the manager reports only its own positions. |
| `table_entry_total` | `int64` | Yes | Total number of holding entries, **as declared on the cover page**. Null for notice filings. |
| `table_value_total` | `int64` | Yes | Total market value of all holdings in **whole US dollars**, as declared on the cover page. Null for notice filings. |

---

## `holdings.parquet`

One row per position. Rows are not individually unique — see *Duplicate entries* below.

| Column | Type | Null? | Description |
|---|---|---|---|
| `accession_number` | `string` | No | Foreign key to `filings.parquet`. |
| `cik` | `string` | No | CIK of the filing manager, zero-padded to 10 characters. Denormalized from the filing for convenience. |
| `report_quarter` | `string` | No | Reporting period as `YYYYQN`. Denormalized from the filing. |
| `name_of_issuer` | `string` | No | Issuer name as filed. Free text, not standardized across filers — the same issuer may be spelled differently by different managers. XML entities must be decoded (`&amp;` becomes `&`). |
| `title_of_class` | `string` | No | Security class as filed (e.g. `COM`, `COM CL A`, `SPONSORED ADS`, `NOTE 2.500% 3/15/29`). Free text. |
| `cusip` | `string` | No | 9-character security identifier. **Must be stored as a string of exactly 9 characters with leading zeros preserved.** Not all CUSIPs are numeric: values beginning with a letter (e.g. `G11448100`) are CINS codes identifying foreign issuers and are valid. |
| `figi` | `string` | Yes | Financial Instrument Global Identifier. An optional field added to Form 13F in 2023; absent from most filings. |
| `value` | `int64` | No | Market value of the position in **whole US dollars**. Filings before Q1 2023 reported this in thousands; filings in scope for this challenge report whole dollars. |
| `ssh_prnamt` | `int64` | No | Quantity held — either a share count or a principal amount, depending on `ssh_prnamt_type`. |
| `ssh_prnamt_type` | `string` | No | Unit for `ssh_prnamt`: `SH` for shares, `PRN` for principal amount. |
| `put_call` | `string` | Yes | Option type where the position is an option, otherwise null. **Note that filers are inconsistent in their capitalisation of this field.** The element is absent — not empty — for non-option positions. |
| `investment_discretion` | `string` | No | Who exercises investment discretion: `SOLE`, `DFND` (defined), or `OTR` (other). |
| `other_manager` | `string` | Yes | Reference to the other manager reporting this position, as filed. Null when the filing manager reports the position alone. Note that this is a reference, not a name or identifier — the cover page defines what it points to. |
| `voting_sole` | `int64` | No | Shares over which the manager holds sole voting authority. |
| `voting_shared` | `int64` | No | Shares over which voting authority is shared. |
| `voting_none` | `int64` | No | Shares over which the manager holds no voting authority. Note that the source XML element for this field is named `None`. |

---

## Duplicate entries

`cusip` is **not** unique within a filing. Do not deduplicate, aggregate, or collapse rows on it, or on any combination of columns. A single filing may legitimately contain multiple entries sharing a CUSIP, and every entry must be preserved as its own row.

Row order within a filing should follow the order of the source information table.

---

## Namespaces

Filers use different namespace prefixes for the same schema. One filing may declare a default namespace, another may bind the identical namespace URI to a prefix such as `ns1:`. Match elements on namespace and local name rather than on the literal tag string.

---

## File conventions

- Format: Parquet, Snappy compression.
- Write to `output/filings.parquet` and `output/holdings.parquet`.
- Partitioning is optional; if you partition, `report_quarter` is the sensible key.
- Column names exactly as specified, lowercase snake_case.
- Use the Arrow types given above. Where a type is nullable, use a genuine null rather than a sentinel value such as `""`, `0`, or `"N/A"`.
