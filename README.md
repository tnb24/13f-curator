# 13F Curator

[![ci](https://github.com/tnb24/13f-curator/actions/workflows/ci.yml/badge.svg)](https://github.com/tnb24/13f-curator/actions/workflows/ci.yml)

13F Curator builds a dataset of 13F holdings from SEC EDGAR for a list of investment
managers. An LLM agent answers questions about the dataset with SQL and cites the
filings it used.

Form 13F is the quarterly report of stock holdings that large investment managers
file with the SEC.

## Quick start

Python 3.12.

```bash
pip install -r requirements.txt
cp .env.example .env   # set LLM_BASE_URL to an OpenAI-compatible endpoint
```

The dataset is already in `output/`, so the agent runs without the pipeline:

```bash
python -m agents.answer "Which manager held the largest Apple position in 2026 Q2?"
# {"answer": "AQR Capital Management LLC", "unit": "NAME", "sources": ["0001167557-26-000226"]}
```

Add `--trace` to see the resolved names, the SQL, and the timings.

To rebuild the dataset from EDGAR, give a User-Agent with a contact email. SEC
requires one.

```bash
python main.py --user-agent "Your Name you@example.com"
python main.py --user-agent "..." --filers my_managers.csv   # your own managers
```

## The dataset

`filers.csv` lists 20 managers by name. A `cik` column is optional. The pipeline
collects every 13F that they filed for 2026 Q1 and Q2, up to 2026-08-18. It writes
`output/filings.parquet` and `output/holdings.parquet` to the schema in
[docs/SCHEMA.md](docs/SCHEMA.md).

| Measure | Value |
|---|---|
| Filings | 40 |
| Holdings | 134,635 rows |
| Schema check (`python verify.py`) | 8 of 8 pass |
| Rerun | same bytes in 85 of 86 output files (the manifest records run times); all 183 requests read from the local store |

```
filers.csv
  -> curator/cik.py           find each manager's SEC filer by name
  -> curator/discover.py      list the manager's 13F filings for the two quarters
  -> curator/download.py      download the XML of each filing
  -> curator/parse.py         parse the XML into Arrow tables, one process per filing
  -> curator/write.py         write the two Parquet tables
  -> curator/attribution.py   find holdings that one manager reports in another's filing
  -> curator/cusip_check.py   check each Q2 CUSIP against the SEC official 13F list
```

- The pipeline finds each manager by name, not by CIK, the ID that SEC gives each
  filer. A hand-typed CIK can be stale or belong to another filer. In the bundled
  list, 2 of 20 do. `output/filers.csv` shows each match and whether the input CIK
  agreed.
- The pipeline sends one request at a time, at 5 per second. SEC allows 10. It saves
  each response under the hash of its URL in `.cache/http`. A rerun then needs no
  network and writes the same output.

## The agent

```
question
  -> scope check       a question only about other quarters returns null, with no model call
  -> name resolution   find the funds and companies that the question names
  -> model call        write one SELECT statement
  -> validator         one statement, known tables and columns, a citation column
  -> DuckDB            run it on read-only views of the two Parquet tables
  -> {"answer", "unit", "sources"}
```

- The agent finds a company by its CUSIP, the nine-character ID of a security, not by
  its name. Filers spell Apple as `APPLE INC`, `APPLE   INC`, and `Apple Inc - US`.
  A filter on `APPLE INC` misses 12 of the 85 Apple rows.
- A sparse word matrix, weighted by how rare each word is, scores the names. A typo
  such as "Nvdia" still finds NVIDIA.
- If the validator rejects the SQL, or DuckDB cannot bind it, the model gets one
  retry with the error.
- The validator and the database each block writes and file access, and neither
  depends on the prompt. `python scripts/demo_security.py` shows both.

### Accuracy

An answer key of hand-checked reference SQL grades each answer ([evals/](evals/)).
Each question ran 3 times with `google/gemma-4-31b-it` at temperature 0. "First
version" is the agent at commit `d262efd`, with the same grader and model.

| Question set | First version | Now |
|---|---|---|
| Development, 23 questions | 43 of 69 | 67 of 69 |
| Held-out, 12 questions | 24 of 36 | 36 of 36 |
| Citation recall / precision | 0.64 / 0.68 | 0.98 / 0.98 |

- The development set shaped the prompt, so its score is optimistic.
- No result on the held-out set changed the prompt or the code. Twelve questions is
  a small sample.
- Scores change between runs, even at temperature 0. Other development runs scored
  68 and 69 of 69.

### Speed

Both versions ran in one session on one laptop. Method and evidence:
[docs/PERFORMANCE.md](docs/PERFORMANCE.md).

| Measure | First version | Now | Change |
|---|---|---|---|
| Rebuild the dataset from the local store | 19.2 s | 4.5 s | 4.3× |
| Agent work per question, model excluded | 2,751 ms | 18 ms | 150× |
| Question to answer, live model | 5.0–5.1 s | 1.9–2.0 s | 2.6× |

Most of the gain comes from loops rewritten as column operations in numpy and
Arrow. No gain comes from a cache of computed results.

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest -q                 # 271 tests; no network, no model
python verify.py                    # check the tables against docs/SCHEMA.md
python -m evals.run --set holdout   # grade the agent; needs a model endpoint
python scripts/bench.py             # time the pipeline and the agent
python scripts/eda.py               # explore the raw XML
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
main.py       pipeline entry point
curator/      fetch, parse, and write the dataset
agents/       the question-answering agent
evals/        answer keys, grader, and recorded runs
scripts/      exploration, benchmarks, security demo
tests/        unit and end-to-end tests; tests/world.py is a fake EDGAR
docs/         schema, example questions, performance notes
filers.csv    the 20 bundled managers
output/       the dataset
verify.py     schema checker
```

## Limitations

- The prompt sets fixed conventions. For example, a "position" means shares held
  directly. An option row reports the value of the underlying shares, so the agent
  leaves options out unless the question asks for them.
- The agent knows only the names that filers use. "Google" does not find Alphabet.
- The validator checks that the SQL is safe, not that it answers the question. The
  evals measure that; nothing checks it at run time.
- I tested one model on one provider.
- The query timeout stops DuckDB, but it does not limit memory or CPU.
- The quarters and the filing cutoff are constants in `curator/scope.py`.

## License

[MIT](LICENSE)

> **Note:** The challenge organisers supplied `agents/llm.py`, `verify.py`, `filers.csv`,
> `docs/SCHEMA.md`, and the example questions.
