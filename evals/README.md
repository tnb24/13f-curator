# Evaluation records

The question sets contain reference answers and citation sets for the bundled
2026 Q1–Q2 dataset. `questions.json` is the development set; `holdout.json` is
the original holdout set. The development set has since grown from 23 to 27
questions. The original holdout has been run repeatedly and its failures have
been inspected, so it should not be treated as a fresh, unseen test set.

## Historical multi-model runs

The four `*-qwen-qwen3-30b-a3b-instruct-2507.json` and
`*-meta-llama-llama-3.3-70b-instruct.json` files are earlier OpenRouter runs,
made on November 3, 2026, before the current lookup and validation changes.
The original session shows the runs were launched from `main` near commit
[`58abefd`](https://github.com/tnb24/13f-curator/commit/58abefd) with a clean
tracked worktree at the start. The individual result files do not record a
commit or tree hash, so the precise source state for each request cannot be
independently established from the files alone.

Each model was asked the same 23 development and 12 holdout questions three
times, at temperature 0. The table counts answers, not distinct questions.
Citation recall and precision are macro-averaged across answers.

| Model | Development | Original holdout | Dev citation recall / precision | Holdout citation recall / precision |
|---|---:|---:|---:|---:|
| [Gemma 4 31B](results/dev-google-gemma-4-31b-it.json) | 67 / 69 | 36 / 36 | 0.971 / 0.971 | 1.000 / 1.000 |
| [Llama 3.3 70B](results/dev-meta-llama-llama-3.3-70b-instruct.json) | 56 / 69 | 33 / 36 | 0.899 / 0.899 | 0.972 / 0.972 |
| [Qwen3 30B-A3B](results/dev-qwen-qwen3-30b-a3b-instruct-2507.json) | 52 / 69 | 27 / 36 | 0.942 / 0.942 | 0.917 / 0.917 |

The corresponding holdout traces are available for
[Gemma](results/holdout-google-gemma-4-31b-it.json),
[Llama](results/holdout-meta-llama-llama-3.3-70b-instruct.json), and
[Qwen](results/holdout-qwen-qwen3-30b-a3b-instruct-2507.json). One failed Qwen
development answer followed an endpoint error; the scores include such errors,
as well as incorrect answers and SQL failures. The results are small-sample
historical observations, not a ranking or a measure of the current agent.

The four recovered files are kept unchanged. Their questions, per-answer
grades, citation scores and summary totals are checked against the answer key
by `tests/test_eval_records.py`.

## Current-agent runs

The `merged-agent-*-gemma-single.json` files record one later run per question
after lookup and validation changes: 27 development and 12 original holdout
questions. See the [project README](../README.md#accuracy) for those results.
As with the earlier files, the records do not embed an exact source tree hash.

To create a new record without overwriting an existing one:

```bash
python -m evals.run --set dev --repeat 3 --out my-dev-run.json
python -m evals.run --set holdout --repeat 3 --out my-holdout-run.json
```

These commands need a configured model endpoint and can incur provider charges.
They write answer traces locally. The live model, provider, question set, code
version, data, and decoding settings all affect the results.
