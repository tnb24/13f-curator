# 13F Curator

[![ci](https://github.com/tnb24/13f-curator/actions/workflows/ci.yml/badge.svg)](https://github.com/tnb24/13f-curator/actions/workflows/ci.yml)

A pipeline that builds a clean, typed dataset of 13F holdings from SEC EDGAR for a
list of investment managers. An LLM agent then answers questions about the dataset
with SQL, and cites the filings it used.

## What it does

- **Input**: `filers.csv`, a list of investment managers by name. A `cik` column is
  optional.
  - A CIK (Central Index Key) is the ID that SEC EDGAR gives to each filer.
- **Output**: every 13F holding that those managers reported for 2026 Q1 and 2026 Q2,
  as two Parquet tables with a fixed schema ([docs/SCHEMA.md](docs/SCHEMA.md)).
  - Form 13F is the quarterly report of equity holdings that large managers file
    with the SEC.
- **Two extra outputs**: the holdings that a manager reports inside another manager's
  filing, and a check of each CUSIP against the SEC official 13F list.
  - A CUSIP is the nine-character ID of a security.
- **An agent** answers questions about the tables and lists its sources.
- The bundled `filers.csv` holds 20 large managers, and `output/` holds their dataset.

## Results

### The dataset

| Measure | Value |
|---|---|
| Filings | 40 (20 managers × 2 quarters) |
| Holdings rows | 134,635 |
| Managers resolved to SEC filers | 20 of 20 |
| Second run | all 85 data and XML files byte-identical; all 183 requests from the local response store |
| Schema check (`verify.py`) | 8 of 8 pass |
| Tests | 270 pass, with no network and no model |

### The agent

Each answer is graded against an answer key: reference SQL that I checked by hand
against the data ([evals/](evals/)). Each set ran 3 times with
`google/gemma-4-31b-it` at temperature 0.

| Question set | Original agent | This agent |
|---|---|---|
| Development set, 23 questions | 43 of 69 (62%) | 67 of 69 (97%) |
| Held-out set, 12 questions | 24 of 36 (67%) | 36 of 36 (100%) |
| Citation recall / precision, both sets | 0.64 / 0.68 | 0.98 / 0.98 |

- The development set holds 10 example questions from the original challenge and 13
  of mine.
  Its failures shaped the prompt, so it is not a fair test on its own.
- No result on the held-out set changed the prompt or the code. Twelve questions is
  a small sample.
- Temperature 0 does not give the same reply each time. Other runs of the development
  set scored 68 and 69 of 69. Both failures above are one question, `most-rows-q2`,
  where the model wrote an invalid `GROUP BY` twice.
- The original agent ran with the same grader and the same model
  ([evals/baseline.py](evals/baseline.py)).

### Speed

Same machine, same data, one clean virtual environment for both versions, medians.
Method and evidence for each number: [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

| Measure | Original | This version | Change |
|---|---|---|---|
| Rebuild the dataset from the local response store | 19.2 s | 4.5 s | 4.3× |
| Rebuild, first run with no derived files | 30.0 s | 4.8 s | 6.3× |
| Agent work per question, model excluded | 2,751 ms | 18 ms | 150× |
| Agent startup, imports excluded | 594 ms | 273 ms | 2.2× |
| Interpreter and agent imports | 2.7 s | 0.74 s | 3.6× |
| Question to answer, live model | 5.0 to 5.1 s | 1.9 to 2.0 s | 2.6× |

- Both versions ran in one session, with their runs interleaved. Absolute times on
  this laptop change by up to 2× between sessions, so compare the ratios.
- With the local work at 18 ms, the model round trip is more than 99% of what is left.

## How it works

### The pipeline

```
filers.csv (manager names)
  -> curator/cik.py           resolve each manager name to its SEC filer
  -> curator/discover.py      list the in-scope filings from the submissions API
  -> curator/download.py      store the XML of each filing under output/filings/
  -> curator/parse.py         read each filing into Arrow columns, in parallel processes
  -> curator/write.py         write output/filings.parquet and output/holdings.parquet
  -> curator/attribution.py   recover holdings that a notice filer reports in another filing
  -> curator/cusip_check.py   check each 2026 Q2 CUSIP against the SEC official list
```

- **Filers resolve by name.** A hand-typed CIK can be stale, or it can belong to
  another filer. So the pipeline never uses an input CIK to find a filer. It matches
  each name against SEC's name-to-CIK file, and it keeps only the candidates that
  filed a 13F in scope. `output/filers.csv` records whether each input CIK agreed.
- The bundled list shows why. Two of its 20 CIKs belong to other live 13F filers,
  and each of those filers has two filings in scope. A count of filings cannot catch
  that, but resolution by name replaces both.

- One HTTP client sends all requests, one at a time, at 5 requests per second.
  [SEC permits 10](https://www.sec.gov/about/developer-resources).
- A response store keeps each EDGAR response under the SHA-256 hash of its URL. A
  rerun reads every response from it and writes the same data files. The store exists
  for SEC's fair-access rules and for reproducibility. No derived file is kept.
- `curator/lookup.py` normalises SEC's one-million-line name-to-CIK file as byte
  arrays, with numpy and Arrow kernels. It agrees with the reference Python
  normaliser on every line of the real file.
- Parse reads each XML row in Python into column lists, then builds typed Arrow
  tables. Each filing runs in its own worker process.

### The agent

```
question
  -> scope check        a question that names only quarters outside 2026 Q1-Q2 returns null
  -> name resolution    funds by name; issuers by CUSIP, which every spelling shares
  -> one model call     question + schema + resolved names -> one SELECT statement
  -> static validator   one statement, known tables and columns, a citation column
  -> read-only DuckDB   views over the two Parquet files; one SELECT, no other files
  -> {"answer", "unit", "sources"}
```

- **Name resolution** holds every name as a row of a sparse name-by-word matrix,
  weighted by how rare each word is. One matrix-vector product scores all 14,394
  issuer spellings in about 1.5 ms, and it accepts a misspelling such as "Nvdia".
- **Issuers resolve to CUSIPs.** Filers spell Apple as `APPLE INC`, `APPLE   INC`, and
  `Apple Inc - US`. A filter on the name `APPLE INC` misses 12 of 85 Apple rows; a
  filter on the CUSIP finds all of them.
- **Repairs**: a statement that the validator rejects, or that DuckDB cannot bind,
  goes back to the model once, with the error.
- **Two guards protect the data**: the validator and the engine. The engine accepts
  one SELECT statement only. Its locked DuckDB settings make the database read-only
  and block every file except the two tables. Neither guard depends on the prompt.
  `python scripts/demo_security.py` shows both.

## Quick start

Python 3.12.

```bash
pip install -r requirements.txt     # the pipeline and the agent
pip install -r requirements-dev.txt # also tests, lint, and scripts/eda.py
cp .env.example .env                # set LLM_BASE_URL to an OpenAI-compatible endpoint
```

The dataset is in `output/`, so the agent runs without the pipeline:

```bash
python -m agents.answer "Which manager held the largest Apple position in 2026 Q2?"
# {"answer": "AQR Capital Management LLC", "unit": "NAME", "sources": ["0001167557-26-000226"]}

python -m agents.answer --trace "..."  # also print the resolved names, SQL, and timings
```

To rebuild the dataset from EDGAR, give a User-Agent with a contact email. SEC
requires it.

```bash
python main.py --user-agent "Your Name you@example.com"
python main.py --user-agent "..." --filers my_managers.csv  # your own list of managers
python verify.py                    # check the Parquet tables against docs/SCHEMA.md
python -m pytest -q                 # run the tests
python -m evals.run --set holdout   # grade the agent against the held-out answer key
python scripts/bench.py             # measure the pipeline and the agent
```

Other scripts:

```bash
python scripts/eda.py            # explore the raw XML (run main.py first)
python scripts/demo_security.py  # show the validator and the read-only engine
```

Docker:

```bash
docker build -t curator .
docker run --rm -v "$PWD/output:/app/output" -v "$PWD/.cache:/app/.cache" \
  curator --user-agent "Your Name you@example.com"
docker run --rm --env-file .env --entrypoint python curator -m agents.answer "..."
```

## Layout

```
main.py          pipeline entry point
curator/         fetch, parse, and write the dataset
agents/          question-answering agent
evals/           answer keys, grader, and recorded runs
scripts/         exploration, benchmarks, security demo
tests/           unit and end-to-end tests; tests/world.py is a synthetic EDGAR
docs/            output schema, example questions, performance notes
filers.csv       the bundled list of 20 managers
output/          the dataset: Parquet tables, quality report, request manifest
verify.py        schema checker
```

## Known limitations

- **Conventions decide some answers.** "Position" means a direct holding: an option
  row reports the notional value of the underlying shares, so the agent leaves options
  out unless the question asks for them. A position grows or shrinks by its share
  count. These rules live in the prompt, and a question that means otherwise gets the
  convention's answer.
- **No alias table.** The resolver matches the names that filers use. "Google" does
  not find Alphabet.
- **No step checks the SQL against the question.** The validator checks that a
  statement is safe, not that it answers the question. The evals measure accuracy;
  nothing enforces it at run time.
- **One model.** I tested one model on one provider. The latency figures include that
  provider's network and queue time.
- **The query timeout interrupts DuckDB only.** It does not cap memory or CPU.
- **Fixed scope.** The quarters (2026 Q1 and Q2) and the filing-date cutoff
  (2026-08-18) are constants in `curator/scope.py`. The cutoff fixes a snapshot of
  EDGAR, so a rerun writes the same bytes.

## Origin

The first version was my entry to the DSRS Data Curator Challenge, a data science
intern screening at the Gies College of Business, University of Illinois, in August
2026. This version adds an answer key, a held-out evaluation, and a data-oriented
rewrite of the slowest paths.

The challenge supplied `verify.py`, `filers.csv`, `docs/SCHEMA.md`, the example
questions, and the first version of `agents/llm.py`. This version of `agents/llm.py`
keeps that module's interface and replaces the OpenAI SDK with httpx.

## License

[MIT](LICENSE)
