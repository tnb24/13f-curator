# 13F Curator

[![ci](https://github.com/tnb24/13f-curator/actions/workflows/ci.yml/badge.svg)](https://github.com/tnb24/13f-curator/actions/workflows/ci.yml)

13F Curator collects investment managers' quarterly holdings from SEC EDGAR.
An LLM agent answers questions about the data using SQL and cites the source filings.

## Get started

Requires Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/demo.py
```

The demo uses the included dataset and a scripted response. It does not call a model.

To ask your own questions, copy `.env.example` to `.env` and configure an
OpenAI-compatible model endpoint. Then run:

```bash
python -m agents.answer "Which manager held the largest Apple position in 2026 Q2?"
```

The response contains an answer, its unit, and SEC filing accession numbers.
Add `--trace` to see name matches, SQL, and timings. An out-of-scope question
returns a null answer. Errors also return a null answer, with details on stderr
and a nonzero exit code.

## Dataset

The included dataset covers 20 managers in `filers.csv` for 2026 Q1 and Q2,
with filings through August 18, 2026. It contains 40 filings and 134,635 holdings.
The tables are `output/filings.parquet` and `output/holdings.parquet`; see the
[schema](docs/SCHEMA.md) for their fields.

To rebuild from EDGAR, provide a User-Agent with a contact email:

```bash
python main.py --user-agent "Your Name you@example.com"
```

Use `--filers my_managers.csv` for another manager list and `--output my-data`
for another output directory. The CSV must have a `fund_name` column; `cik` is
optional. Each manager must resolve to a different SEC filer. Use `--quarters`
and `--cutoff` to change the default reporting period and filing cutoff.

```bash
python main.py --user-agent "Your Name you@example.com" --quarters 2026Q2,2026Q3 --cutoff 2026-11-16
python -m agents.answer --data-dir my-data "How many filings are in 2026 Q2?"
```

The pipeline caches SEC responses in `.cache/http`. If a rebuild fails, rerun it
or use a fresh output directory before relying on its files. The manifest records
the status of optional reports.

## How the agent works

The agent matches fund and company names locally, then sends the question, table
schema, and relevant name matches to the configured model. The model generates
SQL. The agent validates the SQL and runs it locally against read-only DuckDB
views of the Parquet tables. It returns the result with citations. The full dataset
is not sent to the model. A retry may include SQL or an error message.

The agent uses security identifiers to account for variations in company names.
It can ask for one additional name lookup and retry a failed query once. SQL
validation checks safety and citations, but it cannot guarantee that an answer
is correct. Some aliases may not resolve, and model answers should be checked.

One evaluation with `google/gemma-4-31b-it` on OpenRouter answered 26 of 27
development questions and 12 of 12 holdout questions. This was one run on small
sets; the holdout had been evaluated before. See the [evaluation notes](evals/README.md)
and [performance notes](docs/PERFORMANCE.md) for results and methods.

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest -q
python verify.py
```

The tests require no network or model. `verify.py` checks the bundled tables
against the schema. See [evals/](evals/) for evaluation commands and results.

To run with Docker:

```bash
docker build -t curator .
docker run --rm -v "$PWD/output:/app/output" -v "$PWD/.cache:/app/.cache" \
  curator --user-agent "Your Name you@example.com"
docker run --rm --env-file .env --entrypoint python curator -m agents.answer "..."
```

Docker's `--env-file` expects plain `KEY=value` lines without quotes or inline
comments. To reach a model server on the host, use `host.docker.internal` in
`LLM_BASE_URL`. On Linux, also pass
`--add-host=host.docker.internal:host-gateway` to `docker run`.

## License

[MIT](LICENSE)

Thanks to DSRS DataHub for providing the challenge files.
