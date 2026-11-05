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
python -m venv .venv
source .venv/bin/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/demo.py
```

The offline walkthrough uses a scripted SQL response to show name resolution,
validation, query execution and citations. It does not call an LLM. The dataset
is already in `output/`, so no pipeline run is needed.

For arbitrary questions, copy `.env.example` to `.env` and configure a running
OpenAI-compatible endpoint and model. Then run:

```bash
python -m agents.answer "Which manager held the largest Apple position in 2026 Q2?"
# {"answer": "AQR Capital Management LLC", "unit": "NAME", "sources": ["0001167557-26-000226"]}
```

Add `--trace` to see the resolved names, the SQL, and the timings.
An out-of-scope question returns a null answer with exit status 0; a model,
database, or startup failure returns a null answer with exit status 1 and an
error on stderr.

To rebuild the dataset from EDGAR, give a User-Agent with a contact email. SEC
requires one.

```bash
python main.py --user-agent "Your Name you@example.com"
python main.py --user-agent "Your Name you@example.com" --filers my_managers.csv --output my-data
python -m agents.answer --data-dir my-data "How many filings are in 2026 Q2?"
python main.py --user-agent "Your Name you@example.com" --quarters 2026Q2,2026Q3 --cutoff 2026-11-16
```

The agent reads the bundled `output/` by default. Pass `--data-dir` to query
another directory containing `filings.parquet` and `holdings.parquet`.
Your managers CSV needs a `fund_name` header and may also include `cik`; see
[`filers.csv`](filers.csv) for an example. Each manager must resolve to a
different SEC filer. Duplicate names or aliases resolving to the same CIK are
rejected to prevent duplicate holdings.

The command exits with status 1 if an optional attribution or CUSIP report fails.
The core tables remain available, the failed report is removed, and the new
`manifest.json` records each report's status. A rebuild that fails before the
manifest is written may leave a mixture of old and new files; use a fresh output
directory or rerun it successfully before relying on those files.

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
| Cached rebuild | six published data and report files matched byte for byte; all 179 requests read from the local store |

```
filers.csv
  -> curator/cik.py           find each manager's SEC filer by name
  -> curator/discover.py      list the manager's 13F filings for the two quarters
  -> curator/download.py      download the XML of each filing
  -> curator/parse.py         parse the XML into Arrow tables using worker processes
  -> curator/write.py         write the two Parquet tables
  -> curator/attribution.py   find holdings that one manager reports in another's filing
  -> curator/cusip_check.py   check each Q2 CUSIP against the SEC official 13F list
```

- The pipeline finds each manager by name, not by CIK, the ID that SEC gives each
  filer. A hand-typed CIK can be stale or belong to another filer. In the bundled
  list, 2 of 20 do. `output/filers.csv` shows each match and whether the input CIK
  agreed.
- The pipeline sends one request at a time, at 5 per second. SEC allows 10. It saves
  each response under the hash of its URL in `.cache/http`. A rerun can use those
  cached responses when all required entries are present. The manifest records
  run-specific timing and cache statistics.

## The agent

```
question
  -> scope check       recognized missing quarters return null, with no model call
  -> name resolution   find the funds and companies that the question names
  -> model call        write a SELECT statement or request a name lookup
  -> validator         check SQL safety, dataset literals, quarters and citations
  -> DuckDB            run it on read-only views of the two Parquet tables
  -> {"answer", "unit", "sources"}
```

- The agent finds a company by its CUSIP, the nine-character ID of a security, not by
  its name. Filers spell Apple as `APPLE INC`, `APPLE   INC`, and `Apple Inc - US`.
  A filter on `APPLE INC` misses 12 of the 85 Apple rows.
- A sparse word matrix, weighted by how rare each word is, scores the names. A typo
  such as "Nvdia" still finds NVIDIA.
- If validation or execution fails, or the result lacks an answer or citation,
  the model gets one retry with the error. It can also request one name lookup.
- The validator and the database each block writes and file access, and neither
  depends on the prompt. `python scripts/demo_security.py` shows both.

### Accuracy

An answer key of hand-checked reference SQL grades each answer ([evals/](evals/)).
One run of the current agent with `google/gemma-4-31b-it` on OpenRouter answered
26 of 27 development questions and 12 of 12 questions in the existing holdout
set. Citation recall and precision were both 1.0 for this run. The development
miss returned the filing's manager name instead of the roster name. These are
single-run results on small sets, and the holdout has been evaluated before;
they do not guarantee correctness on new questions. Full traces:
[development](evals/results/merged-agent-dev-gemma-single.json) and
[holdout](evals/results/merged-agent-holdout-gemma-single.json).

The comparison below predates the current lookup and validation changes.
Each question ran 3 times with `google/gemma-4-31b-it` at temperature 0. "First
version" is the agent at commit `d262efd`, with the same grader and model.

| Question set | First version | Later version |
|---|---|---|
| Development, 23 questions | 43 of 69 | 67 of 69 |
| Held-out, 12 questions | 24 of 36 | 36 of 36 |
| Development citation recall / precision | — | 0.971 / 0.971 |

- The development set shaped the prompt, so its score is optimistic.
- The original holdout has been evaluated repeatedly and its failures inspected.
  Twelve questions is a small sample, not a fresh test of the current agent.
- Scores change between runs, even at temperature 0. Other development runs scored
  68 and 69 of 69.

Earlier three-run evaluations of Llama 3.3 70B and Qwen3 30B-A3B, including
their traces, methodology and caveats, are in the [evaluation notes](evals/README.md).
They predate the current agent and do not establish its accuracy with those models.

### Speed

Both historical versions ran in one session on one laptop; the current changes
have not been benchmarked. Method and evidence:
[docs/PERFORMANCE.md](docs/PERFORMANCE.md).

| Measure | First version | Later version | Change |
|---|---|---|---|
| Rebuild the dataset from the local store | 19.2 s | 4.5 s | 4.3× |
| Agent work per question, model excluded | 2,751 ms | 18 ms | 150× |
| Question to answer, live model | 5.0–5.1 s | 1.9–2.0 s | 2.6× |

Most of the gain comes from loops rewritten as column operations in numpy and
Arrow. No gain comes from a cache of computed results.

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest -q                 # no network or model required
python verify.py                    # check the tables against docs/SCHEMA.md
python -m evals.run --set holdout --out my-eval.json  # needs a model endpoint
python scripts/bench.py             # time the pipeline and the agent
python scripts/eda.py               # explore the raw XML
```

Evaluation records are not overwritten by default. Choose a new `--out` path,
or pass `--force` if you intend to replace an existing record. `--only` accepts
one or more question IDs for a smaller run.

Docker:

```bash
docker build -t curator .
docker run --rm -v "$PWD/output:/app/output" -v "$PWD/.cache:/app/.cache" \
  curator --user-agent "Your Name you@example.com"
docker run --rm --env-file .env --entrypoint python curator -m agents.answer "..."
```

Docker reads `--env-file` differently from Python's dotenv loader: use plain
`KEY=value` lines, without quotes or inline comments, for values passed to the
container. In particular, a comment after `LLM_MODEL` becomes part of the model ID.
For a model server running on the host, `localhost` inside the container refers
to the container itself. With Docker Desktop, use `host.docker.internal` in the
container's `LLM_BASE_URL` instead. On Linux, add
`--add-host=host.docker.internal:host-gateway` to `docker run`.

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
- The model can request a lookup for an alias such as "Google", but it must
  supply a name that matches the filings; resolution is not guaranteed.
- The model may return a filing manager's name instead of the roster name, so
  equivalent managers can appear under different spellings.
- A recognized bare year requires all four quarters. The scope check recognizes
  several common date formats but is not a complete natural-language date parser.
- The validator checks that the SQL is safe, not that it answers the question. The
  evals measure that; nothing checks it at run time.
- The published current evaluation measures one run with one model and provider.
- `LLM_TOKEN_BUDGET` tracks provider-reported tokens after each response. It is
  not a spending cap: a request can exceed the threshold, and providers may omit
  usage data. Set an appropriate limit with your provider as well.
- The query timeout stops DuckDB, but it does not limit memory or CPU.
- The default quarters and filing cutoff are fixed; choose other values with
  `--quarters` and `--cutoff` when rebuilding the dataset.

## License

[MIT](LICENSE)

> **Note:** The challenge organisers supplied `agents/llm.py`, `verify.py`, `filers.csv`,
> `docs/SCHEMA.md`, and the example questions.
