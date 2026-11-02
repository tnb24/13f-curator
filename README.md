# 13F Curator

A pipeline that turns SEC Form 13F filings into a clean, typed dataset. An LLM agent
then answers questions about the dataset with SQL, and cites the filings it used.

I built this project for the DSRS Data Curator Challenge. The challenge was a data
science intern screening at the Gies College of Business, University of Illinois, in
August 2026.

## The problem

- A researcher supplies a roster of 20 investment managers, with a CIK for each.
  - A CIK (Central Index Key) is the ID that SEC EDGAR gives to each filer.
- Some CIKs on the roster are wrong.
- The researcher wants every 13F holding for 2026 Q1 and 2026 Q2.
  - Form 13F is the quarterly report of equity holdings that large managers file
    with the SEC.
- The output is two Parquet tables that match a fixed schema
  ([docs/SCHEMA.md](docs/SCHEMA.md)).
- An agent answers questions about the tables and lists its sources.

## Results

| Measure | Value |
|---|---|
| Filings | 40 (20 managers × 2 quarters) |
| Holdings rows | 134,635 |
| Roster CIKs corrected | 2 of 20 |
| Rerun output | byte-identical; all 182 requests come from the cache |
| Schema check (`verify.py`) | 8 of 8 pass |
| Tests | 67 pass, with no network and no model |

## How it works

### The pipeline

```
filers.csv
  -> curator/cik.py        match each roster name to an EDGAR entity
  -> curator/discover.py   list the in-scope filings from the submissions API
  -> curator/download.py   store the XML of each filing under output/filings/
  -> curator/parse.py      read the cover page and the holdings into typed records
  -> curator/write.py      write output/filings.parquet and output/holdings.parquet
  -> curator/bonus1.py     recover holdings that a notice filer reports in another filing
  -> curator/bonus2.py     check each 2026 Q2 CUSIP against the SEC official list
```

- One HTTP client sends all requests, one at a time, at 5 requests per second.
  SEC permits 10.
- A disk cache stores each response under the SHA-256 hash of its URL.
- A rerun reads from the cache and writes the same bytes.

### The agent

```
question
  -> scope check            a quarter outside 2026 Q1-Q2 returns null with no model call
  -> name resolution        fuzzy match of fund and issuer names to the stored spellings
  -> one model call         question + schema -> one SELECT statement
  -> static validator       one statement, known tables and columns, no file functions
  -> read-only DuckDB       the connection cannot write or read external files
  -> {"answer", "unit", "sources"}
```

- Two guards protect the data: the validator and the read-only connection.
- Neither guard depends on the prompt. `python scripts/demo_security.py` shows both.

## Problems worth a look

**Two wrong CIKs point at real filers.**

- The roster CIK for Tudor Investment Corp belongs to State of Wisconsin Investment Board.
- The roster CIK for Situational Awareness LP belongs to ARK Investment Management.
- Both wrong entities file two 13Fs in scope, so a count check does not find the error.
- The pipeline matches each roster name, then keeps only candidates that file a 13F in
  scope.
- An exact name match is not sufficient. `TUDOR INVESTMENT CORP` files no 13F; the
  correct entity is `TUDOR INVESTMENT CORP ET AL`.
- The code contains no hard-coded correction.

**XML namespaces vary.** Filers bind different prefixes to one namespace. The parser
matches local element names only.

**Missing values stay visible.**

- An empty element becomes null, never zero and never an empty string.
- When a required field is absent, the parser applies a default and records it in
  `output/quality_report.json`.
- The parser never drops a row. On this corpus the quality report is empty.

**Text stays as filed.** Fields such as `put_call` keep the case of the filer. The agent
compares with `upper()` inside SQL.

**A notice filing has no holdings.**

- Pershing Square filed a 13F-NT notice for 2026 Q2. Its parent company reports the
  holdings.
- `bonus1.py` reads the parent from the notice and finds the parent's filing.
- It maps the notice filer to a sequence number on the parent's cover page.
- It writes the 14 rows (of 15) that carry that number to
  `output/bonus_attributed.parquet`.

**CUSIP check.**

- A CUSIP is the 9-character ID of a security.
- `bonus2.py` parses the fixed-width SEC list and checks 32,176 pairs of filing and
  CUSIP.
- Every pair is on the official list for 2026 Q2.

## Quick start

Python 3.12.

```bash
pip install -r requirements.txt
cp .env.example .env    # set LLM_BASE_URL to an OpenAI-compatible endpoint
```

The dataset is in `output/`, so the agent runs without the pipeline:

```bash
python -m agents.answer "How many distinct issuers did Renaissance Technologies LLC report in 2026 Q2?"
# {"answer": 2898, "unit": "COUNT", "sources": ["0001037389-26-000059"]}
```

To rebuild the dataset from EDGAR, give a User-Agent with a contact email. SEC requires
it.

```bash
python main.py --user-agent "Your Name you@example.com"
python verify.py       # check the Parquet tables against docs/SCHEMA.md
python -m pytest -q    # run the tests
```

Other scripts:

```bash
python scripts/eda.py            # explore the raw XML (run main.py first)
python scripts/run_examples.py   # run docs/questions-examples.md through the agent
python scripts/demo_security.py  # show the validator and the read-only connection
```

Docker:

```bash
docker build -t curator .
docker run --rm -v "$PWD/output:/app/output" -v "$PWD/.cache:/app/.cache" \
  curator --user-agent "Your Name you@example.com"
```

## Layout

```
main.py          pipeline entry point
curator/         fetch, parse, and write the dataset
agents/          question-answering agent
scripts/         exploration, example runs, security demo
tests/           unit and integration tests
docs/            output schema and example questions
filers.csv       input roster of 20 managers
output/          the dataset: Parquet tables, quality report, request manifest
                 (main.py also writes the raw XML here, under output/filings/)
verify.py        schema checker
```

## Known limitations

- Issuer names vary between filers, for example `APPLE INC` and `Apple Inc - US`.
  The agent filters on one spelling, so a total can miss rows. A fix: resolve the
  name to CUSIPs, then filter on CUSIP.
- The agent does not separate share rows from option rows unless the question asks.
- A question about "all of 2026" passes the scope check. The agent then sums Q1 and Q2.
- `sources` lists the filings that the SQL returns. No check confirms that the SQL
  answers the question.

## How I built it

I set the direction and made the design decisions. AI coding agents wrote most of the
code: Claude Code planned and reviewed the work, and OpenCode wrote most modules.

The challenge organisers supplied `agents/llm.py`, `verify.py`, `filers.csv`,
`docs/SCHEMA.md`, and the example questions.

## License

MIT for my code. See [LICENSE](LICENSE). The files from the challenge organisers
stay under their terms.
