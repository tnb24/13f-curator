# 13F Curator

[![CI](https://github.com/tnb24/13f-curator/actions/workflows/ci.yml/badge.svg)](https://github.com/tnb24/13f-curator/actions/workflows/ci.yml)

13F Curator turns SEC filings into a queryable holdings dataset. It downloads and
parses Form 13F filings into Parquet, then uses an LLM to translate questions into
SQL and return answers with filing accession numbers.

The repository includes a snapshot of **40 filings and 134,635 holdings** from
20 managers for 2026 Q1 and Q2, with a filing cutoff of August 18, 2026.

## Try it

Requires Python 3.12. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/demo.py
```

The offline demo uses a scripted SQL response, so it needs no model or network.
It asks which manager held the largest Apple position in 2026 Q2 and returns:

```json
{"answer": "AQR Capital Management LLC", "unit": "NAME", "sources": ["0001167557-26-000226"]}
```

### Ask your own question

Copy `.env.example` to `.env` and configure an OpenAI-compatible model endpoint.
The example includes settings for local servers and OpenRouter.

```bash
python -m agents.answer "Which manager held the largest Apple position in 2026 Q2?"
python -m agents.answer --trace "How many filings are in 2026 Q2?"
```

The answer is JSON on stdout; `--trace` prints name matches, SQL, and timing to
stderr. An out-of-scope question returns a null answer. Failures also return a
null answer, with an error on stderr and a nonzero exit code.

## How it works

1. The pipeline resolves manager names to SEC identifiers, discovers filings,
   downloads their XML, and writes typed filings and holdings tables. It caches
   SEC responses and records provenance and quality information alongside the data.
2. The agent resolves manager and issuer names locally and sends the question,
   schema, and relevant matches to the configured model. It does not send the full
   dataset; a retry can include generated SQL and an error message.
3. The model produces SQL, which is validated and run against read-only DuckDB
   views over the Parquet files. The result includes accession numbers for the
   underlying filings. The agent can request an extra name lookup and retry a
   rejected query once.

The engine restricts file access and accepts a single SELECT statement; queries
also have a timeout and row limit. These checks reduce the risk of executing
model-generated SQL, but they do not establish that an answer is correct. Check
important results against the cited filings. Form 13F is a periodic disclosure,
not a complete or real-time view of a manager's portfolio.

## Rebuild the dataset

Provide a User-Agent containing your name and contact email for SEC requests:

```bash
python main.py --user-agent "Your Name you@example.com"
```

The default roster is [filers.csv](filers.csv). A custom CSV needs a `fund_name`
column and can include `cik`; each manager must resolve to a different SEC filer.
Choose another roster, output directory, quarter range, or filing cutoff with:

```bash
python main.py --user-agent "Your Name you@example.com" --filers my_managers.csv --output my-data --quarters 2026Q2,2026Q3 --cutoff 2026-11-16
python -m agents.answer --data-dir my-data "How many filings are in 2026 Q2?"
```

The core outputs are `filings.parquet` and `holdings.parquet`, joined by accession
number. The pipeline also writes a manifest, a quality report, and optional
attribution and CUSIP reports. See the [schema](docs/SCHEMA.md) for field definitions.
The bundled snapshot uses whole-dollar values; filings before 2023 used different
reporting units, so do not assume the same interpretation for older periods.
If a rebuild fails, rerun it or use a fresh output directory before relying on
its files; the manifest records whether optional reports succeeded.

## Accuracy

One run of `google/gemma-4-31b-it` on OpenRouter answered **26 of 27 development
questions** and **12 of 12 holdout questions** correctly. These are small sets,
and the holdout had already been evaluated and inspected, so it is not an unseen
test. Results depend on the model and dataset. The [evaluation notes](evals/README.md)
include the methods and saved runs; [performance notes](docs/PERFORMANCE.md)
describe the measured pipeline and agent optimizations.

## Development

```bash
pip install -r requirements-dev.txt
ruff check .
python -m pytest -q
python verify.py
python scripts/demo_security.py
```

Tests run without a model or network. `verify.py` checks the bundled tables'
structure and integrity; it does not verify the values against the SEC.

The pipeline can also run in Docker:

```bash
docker build -t curator .
docker run --rm -v "$PWD/output:/app/output" -v "$PWD/.cache:/app/.cache" \
  curator --user-agent "Your Name you@example.com"
docker run --rm --env-file .env --entrypoint python curator -m agents.answer "How many filings are in 2026 Q2?"
```

Docker's `--env-file` expects plain `KEY=value` lines without quotes or inline
comments. For a model server on the host, use `host.docker.internal` in
`LLM_BASE_URL`; on Linux, also pass `--add-host=host.docker.internal:host-gateway`.

## License

[MIT](LICENSE). Challenge files provided by DSRS DataHub.
