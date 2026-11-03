# Performance

This document records each speed change: what it claims, how I measured it, the
evidence that the output did not change, and the risk it accepts. It also lists
the attempts that I rejected.

The design rule is data-oriented design (DOD): keep data in contiguous columns,
and transform a whole column with one kernel. A kernel is one compiled function
that processes a whole array. No speed change uses a derived cache. Where a path
still builds Python objects per row, this document says so.

## Summary

| Measure | Before (`d262efd`) | After | Change |
|---|---|---|---|
| Rebuild the dataset from the response store, in process | 19.2 s | 4.5 s | 4.3× |
| Rebuild, first run with no derived files | 30.0 s | 4.8 s | 6.3× |
| Agent work per question, model excluded, median | 2,751 ms | 18.4 ms | 150× |
| Agent work per question, model excluded, p95 | 4,637 ms | 28.9 ms | 160× |
| Agent startup: database and name indexes | 594 ms | 273 ms | 2.2× |
| Interpreter and agent imports | 2.69 s | 0.74 s | 3.6× |
| Question to answer with the live model, median | 5.0 to 5.1 s | 1.9 to 2.0 s | 2.6× |

All figures come from one session on 2026-11-02, with runs of both versions
interleaved. Absolute times on this laptop change by up to 2× between sessions.
An earlier session measured the same rebuild at 10.4 s and 3.0 s. Compare the
ratios, not seconds from different sessions.

## Method

- **Machine.** One Windows 11 laptop on mains power, 16 logical CPUs, Python 3.12.10.
- **Environment.** One new virtual environment for both versions: `requirements.txt`
  plus `openai==2.52.0`, which the original imports. It has no pandas (see P7).
  Both versions use DuckDB 1.5.5 and PyArrow 17.0.0.
- **Data.** Both versions read the same response store, `.cache/http`. The harness
  replaces the network fetch with a function that fails, so no run can make a
  request. Every run shows zero cache misses.
- **Repeats.** Six interleaved rebuilds per version. The first rebuild of the
  original writes its derived lookup copy, so the table uses the five later runs.
  Five more runs per version, each in a new directory, give the "first run" row.
- **The agent without the model.** Each development question's reference SQL
  replaces the model (21 questions). The current agent ran 20 rounds through
  `scripts/bench.py`. The original ran 5 rounds, with `agents.sql.complete` replaced.
  A failed reference query stops the run, so an error cannot count as a fast answer.
- **Startup.** The time to build the database and the name indexes, imports
  excluded. For the original, each sample writes its database copy to a new file.
- **The agent with the model.** `python -m evals.run` and `evals/baseline.py`, three
  times per question. The records are in [evals/results/](../evals/results/).

Reproduce (on Unix, use `bin/` in place of `Scripts/`):

```bash
python -m venv venv-runtime
venv-runtime/Scripts/python -m pip install -r requirements.txt openai==2.52.0
venv-runtime/Scripts/python main.py --user-agent "Name you@example.com"  # fill the store once
venv-runtime/Scripts/python scripts/bench.py --pipeline 5 --agent 20
python -m pytest -q   # every equivalence claim below has a test
```

For the original, check out `d262efd` into a separate folder with `git worktree add`.
Give that folder the same `.cache/http`, and run its `main.py` with the same
interpreter. Its manifest records `elapsed_seconds` for the run.

## Output identity

- All 85 deterministic files are byte-identical to the output of `d262efd`, on every
  run. They are the two Parquet tables, the two bonus files, the corrected roster,
  the quality report, and 79 XML files.
- `manifest.json` is not identical, because it records a start time and elapsed times.
  On a warm run the original also makes 182 requests, not 183: its derived lookup copy
  replaces one request.
- Identity holds for this response store. A new EDGAR response can hold input that
  the tests do not cover.

## Pipeline

### Where the time went

Stage medians, in seconds, from the five interleaved warm runs:

| Stage | Before | After |
|---|---|---|
| verify | 4.58 | 2.52 |
| discover | 0.03 | 0.03 |
| download | 2.40 | 0.25 |
| parse | 10.04 | 1.05 |
| write | 1.06 | 0.18 |
| bonus1 | 0.02 | 0.02 |
| bonus2 | 0.46 | 0.22 |

Component medians, in seconds, from five repeats each:

| Component | Before | After |
|---|---|---|
| Parse the raw name-to-CIK lookup file | 8.94 | 1.36 |
| Fuzzy search for one roster name (Tudor) | 1.37 | 1.37 |
| Read the root tag of every stored XML file | 2.07 | 0.05 |
| Build the HTTP client on a run with no request | 0.42 | 0.00 |
| Import `curator.parse`, interpreter included | 0.49 | 0.36 |
| Import `agents.llm`, interpreter included | 2.02 | 0.38 |

### P1. The name-to-CIK lookup works on bytes

`curator/lookup.py`

- **Before.** The original parsed SEC's 40 MB lookup file (1,056,222 lines) one line at
  a time, with a regular-expression normaliser per name: 8.9 s. To hide that cost it
  wrote a derived copy, `.cache/cik_lookup.csv`, and read it back on later runs: 5.4 s.
- **After.** The derived copy is gone. Each run parses the raw file in 1.4 s:
  - numpy finds every line break and every colon in one pass over the bytes. A line
    break is each byte that `str.splitlines` accepts in Latin-1: `\n`, `\r`, `\v`,
    `\f`, `\x1c` to `\x1e`, and `\x85`;
  - `searchsorted` maps the last two colons of each line to its CIK span;
  - an int8 span mask selects the name bytes into one compact buffer;
  - numpy uppercases the buffer, drops punctuation, drops a leading `THE `, and
    collapses runs of spaces;
  - Arrow's RE2 kernels remove parentheses and `/MA`-style suffixes, and replace `&`
    with ` AND `, on the lines that hold `(`, `/`, or `&` only.
- **Simple lines.** The byte path handles 99.7% of lines: printable ASCII, no leading
  space, and a space after every `&`. On such a line each byte rule gives exactly the
  result of the reference `normalise`. The other 2,661 lines go through `normalise`
  itself.
- **Counting rare flags.** A per-line count of a rare byte (a non-ASCII byte, a `(`)
  starts from the flagged positions: `flatnonzero`, then `searchsorted`, then
  `bincount`. It does not make a running sum over all 40 million bytes.
- **Fuzzy search.** `rapidfuzz.process.cdist` scores the remaining candidates on every
  core. A length bound drops names that cannot reach the threshold: the score is
  `100 × (1 − d / (la + lb))`, and the edit distance `d` is at least `|la − lb|`.
  The bound measures the target after its whitespace collapses, as the scorer does.
  Each hit is then scored again on its own, so its score is the exact float the
  original used. Ties break on the name, not on file order.
- **Evidence.**
  - `tests/test_lookup.py` gives the raw bytes of the real file to both paths: all
    1,044,809 (name, CIK) pairs agree, in both directions.
  - The same file compares the paths on 27 hand-picked hazards, and on 20,000
    random lines over the alphabet `ABCTHEthe ()/.,'"&:;-0-9é`.
  - It also compares them on 1,000 random bodies over all 256 Latin-1 bytes. Other
    cases are each of the nine line breaks, and an empty body.
  - `filers.csv` is byte-identical.
- **Risk.** The byte path is exact only for "simple" lines. A future change to a byte
  rule must keep the random tests green. The tests run on every commit.

### P2. The root-tag check reads one tag

`curator/download.py`, `root_localname`

- **Before.** To learn whether a file is an information table or a cover page, the
  original parsed the whole document, 13 MB for the largest. All stored files: 2.07 s.
- **After.** `iterparse` stops at the first start tag: 0.05 s. Download: 2.40 s to 0.25 s.
- **Evidence.** The stored files and both tables are byte-identical.
- **Risk.** A file that is malformed after its first tag now fails in parse, loudly. The
  original marked it "not XML" and could treat a broken information table as a notice.

### P3. Parse fills columns, in parallel processes

`curator/parse.py`

- **Before.** One Python dict per holding row (134,635 rows), six child-map builds per
  row, and `etree.QName` for every tag. The writer then built a second dict per row.
  Parse: 10.04 s.
- **After.** Parse: 1.05 s.
  - Each field goes straight into its column list, and each filing returns an Arrow
    table.
  - The tag split is `str.rpartition`: 0.02 s per 100,000 tags, against 0.11 s for
    `QName`.
  - Each row builds three child maps, not six.
  - One filing is one task in a process pool. Its workers start at the beginning of
    the run and import lxml and pyarrow while the main process verifies CIKs. The
    largest files go first.
- **Not columnar.** The parser still walks each holding element in Python. Each row
  builds three child maps and appends Python values to the column lists. Only the
  step from lists to Arrow is columnar. The processes spread the Python work across
  cores.
- **Why processes.** lxml creates one Python proxy object for each element that Python
  touches, and that work holds the GIL. Threads therefore cannot run the row walk in
  parallel.
- **Evidence.**
  - Both tables are byte-identical to the original output.
  - `tests/test_pipeline.py::test_parallel_parse_writes_the_same_bytes` runs the
    pipeline with one worker and with two, and compares every output file.
  - `tests/test_determinism.py::test_chunk_layout_does_not_change_the_bytes` shows
    that one table per filing, concatenated, writes the same Parquet bytes as one
    table. The writer combines chunks first.
- **Risk.** Eight workers hold eight lxml trees at once. The largest tree is a 13 MB
  file. On Windows each worker starts a new interpreter, so `main.py` imports nothing
  at module level (see P4).

### P4. Workers import only what they use

- **Before.** `curator.parse` imported `curator.download`, which imported discover,
  cik, numpy, rapidfuzz, and httpx. On Windows a spawned worker also re-imports
  `main.py`. The import took 0.49 s, interpreter included.
- **After.** The shared records moved to `curator/records.py`, which imports only the
  standard library. The import takes 0.36 s, most of it pyarrow and lxml. `main.py`
  imports the pipeline inside its `__main__` guard only.

### P5. The writer stores typed tables

`curator/write.py`

- **Before.** A projection function per row, then `pa.Table.from_pylist`: 1.06 s.
- **After.** The tables arrive typed, so the writer checks the schema and writes:
  0.18 s. A table with a wrong schema never reaches disk
  (`test_a_table_with_the_wrong_schema_never_reaches_disk`).

### P6. No TLS context on a cached run

`curator/http.py`. The HTTP client opens on the first network request. A run that
reads every response from the store no longer builds a TLS context: 0.42 s saved.

### P7. pandas is not a runtime dependency

When pandas is installed, the first `pa.array()` call in each process imports it.
DuckDB's Python client does the same on the first bound parameter. Nothing at run time
uses pandas, so it moved to `requirements-dev.txt`. The validator passes the statement
as an escaped literal, not a bound parameter, so a development install does not pay
the import there.

- **Evidence.** `python -c "import pyarrow as pa; pa.array([1])"`, median of seven
  processes: 1.14 s with pandas installed, 0.35 s without.
- **Not fixed.** When pandas is absent, pyarrow 17 tries the import again on each
  conversion of a Python value. Each failed attempt is cheap. A fix needs an entry in
  `sys.modules`, or a pyarrow upgrade. An upgrade changes the `created_by` field of
  each Parquet file, so the output is no longer byte-identical to the original.

### P8. Bonus 2 works on columns

`curator/bonus2.py`

- **Before.** A Python dict per holding row (68,070 rows), a dict per (filing, CUSIP)
  pair, a line loop over each official list, and `csv.DictWriter`: 0.46 s.
- **After.** 0.22 s:
  - numpy splits each official list into an Arrow string array, with no Python
    string per line;
  - Arrow kernels slice the fixed-width fields, and `np.unique` keeps the first line
    for each CUSIP;
  - an ordered `group_by` builds the pairs, and `index_in` matches them against the
    list;
  - Arrow kernels render the CSV, and one string buffer holds the whole body.
  - Python code runs once per unmatched pair only.
- **Exactness.** The kernels run only on printable ASCII. There, the space is the only
  whitespace, and one byte is one character. Any other body or column goes through
  the reference rules, `parse_official` and `normalise`. The CSV kernels handle
  string, integer, and boolean columns, and a null becomes an empty field. A table
  with any other column type goes through `csv.writer`, because Arrow and Python
  format floats differently.
- **Evidence.** The CSV is byte-identical. `tests/test_bonus.py` compares each fast
  path with its reference. The cases are 18 list hazards, 2,000 random lists, 300
  random Latin-1 bodies, and 3,003 random CUSIPs. It also compares random tables,
  nulls, quoted headers, and floats against the `csv` module.
- **Risk.** A change to the CSV quoting rules of the `csv` module breaks the match.
  The random test then fails.

### Rejected

I measured these attempts with prototypes that the repository does not keep, so this
table gives no figures.

| Attempt | Why rejected |
|---|---|
| Per-column XPath with an alignment check in libxml2 | slower than the row walk: libxml2 evaluates the predicate XPath slowly |
| stdlib ElementTree in place of lxml | slower: its C parser builds every element first |
| A thread pool for parse | slower than processes: the row walk holds the GIL |
| An Arrow split of the whole lookup text | slower than `str.splitlines`: one 40 MB string costs more to convert |
| A run-length `np.repeat` for the span mask | slower than an int8 running sum |
| Give the parsed submissions JSON from verify to discover | saves milliseconds only, and couples two stages |

### What is left

Verify takes 2.5 s. Two parts, measured on their own, take most of it: the lookup
index (1.4 s) and the one roster name that needs a fuzzy search (Tudor, 1.4 s).
rapidfuzz needs Python strings, so each candidate that passes the length bound
becomes a Python string first.

## Agent

### Where the time went

Per question, the original agent took 2,751 ms at the median. It:

- opened a new DuckDB connection and never closed it;
- scanned every distinct issuer name in Python and called `fuzz.WRatio` once for each
  name and each word group of the question: 14,394 names × up to 60 groups;
- on its first question, built `.cache/curator.duckdb` by copying both tables.

Its imports took 2.7 s, and the OpenAI SDK took most of that.

### A1. One Agent owns the data for its lifetime

`agents/answer.py`, `Agent`. The database and the name indexes live in one object,
built once. A process that asks many questions reuses it. The original kept its name
lists in a process-wide dictionary that nothing ever invalidated.

### A2. Names as a sparse matrix

`agents/resolve.py`, `NameIndex`

- Each name is a row of a CSR (compressed sparse row) matrix over its content words,
  weighted by IDF (inverse document frequency), so "pershing" counts for more than
  "capital".
- A question becomes a sparse vector over the same words. One `np.add.reduceat` over
  the matrix scores every name: 1.5 ms for "nvidia" against 14,394 issuer spellings.
- `fuzz.ratio` runs only against the vocabulary (6,105 words), and only for
  question words of five or more letters, so "Nvdia" still finds NVIDIA.
- The index builds from Arrow string kernels: lower case, non-alphanumeric runs to a
  space, split, filter. An ASCII question takes a scalar path: 0.04 ms, against
  8.5 ms for the kernels on one string. `tests/test_resolve.py`
  compares the two paths on 5,003 strings and on all 16,384 pairs of ASCII characters.
- **Accuracy, not only speed.** Issuers resolve to CUSIPs, which every filed spelling
  shares. The result limit applies after the spellings of each CUSIP merge. So many
  spellings of one CUSIP cannot push out a better CUSIP. The original ranked noise
  first in three known cases: "Nvidia", "Microsoft", and "Tesla".
  `test_real_names_resolve` pins all three.

### A3. The database is two views

`agents/db.py`, `Engine`

- **Before.** The original copied both Parquet files into `.cache/curator.duckdb` and
  checked file times to decide when to rebuild it.
- **After.** Each Engine writes a new database file, in a private temporary directory,
  that holds two views over the Parquet files and no data. It reopens the file
  read-only. `allowed_paths` lists the two Parquet files, external access goes off,
  and the configuration locks. Close deletes the directory.
- **One SELECT only.** The allow list permits writes as well as reads, so `read_only`
  alone does not stop `COPY` to a listed file. `Engine.query` therefore parses each
  statement with DuckDB first, and it accepts one SELECT statement only. It accepts
  no EXPLAIN, because EXPLAIN ANALYZE runs the statement that it explains.
- **Timeout.** A timer interrupts the cursor. DuckDB clears an interrupt when a
  statement starts, so the timer interrupts again every 10 ms until the query ends.
  The timeout interrupts DuckDB only. It does not cap memory or CPU.
- **Cost.** Build and close: 77 ms, against 153 ms for native in-memory tables. A
  query then scans Parquet, not a native table.
- **Evidence.** `tests/test_engine.py` attacks each layer with no validator in front.
  `query` gets every kind of non-SELECT statement. A raw cursor gets writes, schema
  changes, and setting changes. Both get a file next to the tables, path traversal,
  and globs. Other attacks are `COPY` (direct and nested), `EXPLAIN ANALYZE COPY`,
  `ATTACH`, `INSTALL`, `LOAD`, two statements, and a runaway query. The listed file
  keeps its bytes.
- **Risk.** A view reads the file on each query. If the pipeline rewrites the tables
  while an Agent runs, the queries see the new rows and the name index keeps the old
  names until the next Agent starts.

### A4. The startup index needs one scan

`build_resolver` runs one grouped scan of holdings, 63 ms, ordered so that each
CUSIP's most common spelling comes first. That one result gives every spelling, each
CUSIP's usual name, and its usual class.

### A5. The model call goes through httpx

`agents/llm.py`. The agent needs one POST per question. Import of the module took
2.02 s with the OpenAI SDK and takes 0.38 s with httpx, interpreter included. The
module keeps the original interface, settings, budget, mock mode, and the retries on
429 and 5xx with `Retry-After`. `tests/test_llm.py` covers the request body, a retry,
a 400, a reply with no message, the budget cap, and mock mode.

### A6. Fewer and shorter model calls

- `max_tokens` fell from 1,024 to 512. A statement for this schema is about 200 tokens,
  and the cap stops a reply that never closes its JSON.
- A statement that DuckDB cannot bind now goes back to the model once, with the error.
  Before, it ended the question with a null answer.
- Repair calls: 9 in 108 model calls over the final runs of both sets.

### A7. Validation reads the text once per list

- The validator scanned the statement once per forbidden keyword, 25 scans. It now
  uses one compiled alternation per list, so two scans.
- It binds the statement with `DESCRIBE`, which does not run it. An output column
  must hold `accession` in its name and have a string or string-list type. So
  `COUNT(accession_number)` and `* EXCLUDE (accession_number)` fail, because the
  answer would have no citation.
- Validation takes 4.5 ms per question at the median.

### What is left

The local work is about 18 ms per question. The query takes 12 ms, the validator
4.5 ms, and name resolution 1.6 ms. The model round trip is 1.9 to 2.0 s at the
median and 3.9 to 6.2 s at p95. End-to-end times do not show how much of that is the
provider's queue, inference, or the network. A faster provider route, or a local
model, is the next step.

## What was removed, and what stays

| Store | Before | Now |
|---|---|---|
| `.cache/cik_lookup.csv`, derived copy of the lookup file | kept between runs | gone; P1 parses the raw file in 1.4 s |
| `.cache/curator.duckdb`, copy of both tables | kept between runs, rebuilt on file time | gone; A3 builds views per Engine |
| `agents/resolve.py` process-wide name dictionary | never invalidated | gone; A1 owns the index |
| `.cache/http`, raw EDGAR responses | kept | kept |
| `output/filings/` and `.cache/bonus1/`, raw XML copies | kept | kept |

The raw response store stays. SEC's fair-access rules ask clients not to fetch the
same document again, and the byte-identical rebuild depends on fixed inputs. The
store and the XML copies hold inputs, not derived results. Download reads an existing
XML copy and skips a rewrite of identical bytes. It still classifies and parses each
file.
