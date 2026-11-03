"""The two output tables, as Arrow schemas. docs/SCHEMA.md is the contract.

Parse builds columns against these schemas, write stores them, and the bonus
stages read them. One definition serves all three. The column order matches
docs/SCHEMA.md, and the nullable flag is part of the contract: verify.py reads
it back field by field.
"""

from __future__ import annotations

import pyarrow as pa

FILINGS_SCHEMA = pa.schema(
    [
        pa.field("accession_number", pa.string(), nullable=False),
        pa.field("cik", pa.string(), nullable=False),
        pa.field("fund_name", pa.string(), nullable=False),
        pa.field("filing_manager", pa.string(), nullable=False),
        pa.field("form_type", pa.string(), nullable=False),
        pa.field("report_period", pa.date32(), nullable=False),
        pa.field("report_quarter", pa.string(), nullable=False),
        pa.field("filing_date", pa.date32(), nullable=False),
        pa.field("is_amendment", pa.bool_(), nullable=False),
        pa.field("amendment_no", pa.int32(), nullable=True),
        pa.field("amendment_type", pa.string(), nullable=True),
        pa.field("report_type", pa.string(), nullable=False),
        pa.field("form_13f_file_number", pa.string(), nullable=True),
        pa.field("crd_number", pa.string(), nullable=True),
        pa.field("sec_file_number", pa.string(), nullable=True),
        pa.field("other_included_managers_count", pa.int32(), nullable=True),
        pa.field("table_entry_total", pa.int64(), nullable=True),
        pa.field("table_value_total", pa.int64(), nullable=True),
    ]
)

HOLDINGS_SCHEMA = pa.schema(
    [
        pa.field("accession_number", pa.string(), nullable=False),
        pa.field("cik", pa.string(), nullable=False),
        pa.field("report_quarter", pa.string(), nullable=False),
        pa.field("name_of_issuer", pa.string(), nullable=False),
        pa.field("title_of_class", pa.string(), nullable=False),
        pa.field("cusip", pa.string(), nullable=False),
        pa.field("figi", pa.string(), nullable=True),
        pa.field("value", pa.int64(), nullable=False),
        pa.field("ssh_prnamt", pa.int64(), nullable=False),
        pa.field("ssh_prnamt_type", pa.string(), nullable=False),
        pa.field("put_call", pa.string(), nullable=True),
        pa.field("investment_discretion", pa.string(), nullable=False),
        pa.field("other_manager", pa.string(), nullable=True),
        pa.field("voting_sole", pa.int64(), nullable=False),
        pa.field("voting_shared", pa.int64(), nullable=False),
        pa.field("voting_none", pa.int64(), nullable=False),
    ]
)

# Bonus 1 stores parent rows with the notice filer they belong to.
ATTRIBUTED_SCHEMA = pa.schema(
    [*HOLDINGS_SCHEMA, pa.field("attributed_to_cik", pa.string(), nullable=False)]
)
