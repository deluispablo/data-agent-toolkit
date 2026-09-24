# Evaluating csv-inspector against the fixture catalog

`pytest` never measures how accurate the model is: it runs without a model.
Accuracy is measured by `scripts/eval_samples.py`, which runs a real model
against every fixture in `samples/` and scores its answers against the
ground truth in `samples/manifest.json`, and by `scripts/compare_runs.py`,
which puts two or more runs side by side. Both live in the repository only;
they are not part of the installed package.

All commands below run from the repository root with
`uv run --directory agents/csv_inspector ...`, so relative paths such as
`runs/` resolve inside `agents/csv_inspector/`. `runs/` is git-ignored.

## The ritual

Any pull request that could move the numbers (a prompt, grounding,
sampling or default-model change) shows a before/after table:

1. **Baseline**, on `main` (or before your change):

   ```bash
   uv run --directory agents/csv_inspector python scripts/eval_samples.py --repeat 3 --out runs/baseline.jsonl
   ```

2. Make the change.
3. **Candidate**, same flags, new file:

   ```bash
   uv run --directory agents/csv_inspector python scripts/eval_samples.py --repeat 3 --out runs/candidate.jsonl
   ```

4. **Compare**:

   ```bash
   uv run --directory agents/csv_inspector python scripts/compare_runs.py runs/baseline.jsonl runs/candidate.jsonl
   ```

5. Paste the Markdown table and the verdict changes into the pull request.

Use the same model, `--bytes`, `--tail-bytes`, `--timeout` and `--repeat`
for both runs; the summary line records all of them, and the table shows
the model and prompt version of each run so mixed comparisons are visible.
Local runs cost nothing but time; for cloud runs read
[Quota notes](#quota-notes) first.

## What is scored

Each fixture's answer is compared field by field with the manifest's
`expected` block: `encoding` (by canonical codec name, alternatives
allowed), `delimiter`, `quotechar`, `escapechar`, `doublequote`,
`has_header`, `header_row_index`, `footer_lines` (surrounding whitespace
ignored), `footer_rows_to_skip` and `columns`. A field without ground truth
is skipped. A fixture's score is matched / compared; the aggregate score is
the mean over fixtures, leaving out known limitations and fixtures whose
inspection failed (both are reported separately).

`columns` is scored three ways:

- **Exact list match**: the reported names, in order and as written in the
  file (surrounding spaces included), equal the expected list. This is the
  only one that counts in the score.
- **Per-name recall** (diagnostic): the share of expected names the model
  reported, surrounding spaces ignored, so a paraphrase such as `Monto` for
  `Importe` shows up as a partial recall.
- **Count match** (diagnostic): whether the model reported as many columns
  as expected.

With `--repeat N` each fixture runs N times, because answers drift even at
`temperature=0`. Per field, the summary adds the **majority-vote
accuracy** (the field counts as right for a fixture when more than half of
its repeats matched) and the **agreement rate** (the share of repeats that
gave the most common answer; 1.0 means they never disagreed), and lists the
fixtures whose answers differ between repeats. A repeat whose inspection
failed counts as a different answer. Each fixture also gets a verdict,
`pass`, `fail: <fields>` (by majority vote), `error` (every repeat failed)
or `unscored`; `compare_runs.py` lists the fixtures whose verdict changed.

## Flag reference

`eval_samples.py`:

| Flag | Meaning |
|---|---|
| `--backend local\|api` | Backend (default: `LLM_BACKEND`, then local). |
| `--model NAME` | Primary model (default: the backend's configured one). Repeatable: models run one after another, each into its own file. |
| `--fallback-model NAME` | Fallback model (default: the configured one). Skipped when equal to the primary. |
| `--bytes N`, `--tail-bytes N` | Head and tail sample sizes, as in the CLI. |
| `--timeout S` | Time budget per fixture (default 300; 0 disables). |
| `--category C` | Only fixtures of one manifest category. |
| `--fixture NAME` | Only this fixture (repeatable). An unknown name is an error. |
| `--max-fixtures N` | At most N fixtures: the first N by name, after the filters. |
| `--repeat N` | Run each fixture N times (default 1). |
| `--out PATH` | Write a JSONL run. `PATH` is a `.jsonl` file (one model only), a template containing `{model}` (e.g. `runs/{model}.jsonl`), or a directory that receives `<UTC timestamp>-<model>.jsonl`. An existing file is never overwritten. |
| `--keep-raw` | Store every model answer's raw text in the JSONL lines (`raw_response`). |
| `--rpm N` | Never exceed N model requests per minute (the harness sleeps). |
| `--max-calls N` | Hard stop: never more than N model requests in total, over all models. |
| `--dry-run` | Build the prompts, print each fixture's size and the planned call count, call no model. Needs no credentials. |
| `--env-file`, `--no-env-file`, `--log-level` | As in the CLI. |

`compare_runs.py RUN RUN [RUN ...]`: the first run is the reference; columns
are labelled with the file names.

## Run files

`--out` writes one JSON object per line, one line per (fixture, repeat):

```json
{"fixture": "delimiter_semicolon.csv", "category": "delimiter", "repeat": 1,
 "known_limitation": false, "model_used": "qwen2.5-coder:7b",
 "matched": ["encoding", "delimiter"], "mismatched": [{"field": "columns", "expected": ["a"], "actual": ["A"]}],
 "skipped": ["escapechar"], "score": 0.9, "columns_recall": 0.0, "columns_count_match": true,
 "usage": {"model": "qwen2.5-coder:7b", "prompt_tokens": 1450, "completion_tokens": 210,
           "latency_seconds": 4.1, "attempts": 1, "retries": 0, "load_seconds": 0.01,
           "prompt_version": "2026.09-a"},
 "latency_seconds": 4.2, "calls": 1, "error": null, "attempt_errors": {}}
```

`usage` is the inspection's `result.usage` (`null` when it failed; then
`error` and, per model, `attempt_errors` say why). The top-level
`latency_seconds` is measured by the harness around the whole inspection,
failures included. `raw_response` (with `--keep-raw`) is a list of
`{"model", "text"}`, one per model answer in call order.

The last line is `{"summary": {...}}`: harness and prompt versions, backend,
models, `n_bytes`, `tail_bytes`, timeout, repeat, start time, whether the
run stopped early, the aggregate, per-category and per-field scores,
majority-vote scores, agreement, verdicts, token totals and means, latency
p50/p95, model calls, fallback uses, retries, error counts by exception
class (the inspection's and each failed attempt's) and the number of 429
and 503 answers seen in failed attempts.

## Quota notes

The Gemini free tier allows about **20 requests per day per model** and
often answers 503 on structured output. A full catalog run is 80 fixtures,
at least 80 requests per repeat and model; a fixture can need up to four
(primary and fallback, each retried once on 429/503). So:

- Start with `--dry-run` to see the prompt sizes and the planned calls.
- Always pass `--max-calls`. Without it, `--backend api` prints a warning
  with the worst-case call count before starting.
- `--max-calls` counts every request: a successful inspection is charged
  `usage.attempts + usage.retries`; a failed one reports no usage (and a
  failed cloud attempt may hide a retry), so it is charged its worst case,
  one request per candidate model, doubled on the cloud backend. A fixture
  only starts when its worst case still fits, so the run never exceeds the
  limit; it stops early and says so in the report and the summary.
- `--rpm` spaces requests on the harness side (the library itself has no
  rate limiter yet).
- Use `--fixture` (or `--max-fixtures`) to run the [cloud subset](#cloud-subset)
  instead of the whole catalog.

## Adding a fixture: hand-written vs matrix spec

The hand-written fixtures in `samples/generate_samples.py` each isolate one
quirk. Real exports combine them (semicolon + decimal comma + cp1252 +
preamble + totals row) and vary in width and length; those combinations are
specs in `samples/matrix.py`, rendered into `gen_<slug>.csv` files flagged
`"generated": true` in `manifest.json`.

Write a hand-written `SampleCase` in `generate_samples.py` when the fixture
is a readable example of one quirk, or when its bytes need something the
renderer cannot express (a quoted newline, an escape character, mixed line
endings, a model-confusing literal). Its `expected` block is typed by hand.

Add a `FixtureSpec` to `MATRIX` in `samples/matrix.py` when the fixture is a
combination of the renderer's dimensions (delimiter, quote character,
encoding and BOM, line ending, preamble, header, footer kind, width, length,
decimal comma, quoted header, ragged rows). The renderer derives every
expected field from the spec, so the ground truth cannot drift from the
bytes. Values come from a `random.Random` seeded from the slug, so
regeneration is byte-for-byte reproducible.

Either way, run `python generate_samples.py` from `samples/` and commit the
new file together with `manifest.json`. Never edit a fixture by hand.

## Known limitations

Fixtures flagged `"known_limitation": true` in `manifest.json` are expected
to fail today; they are reported apart and left out of every score. The
catalog has **6** (counted from `manifest.json`):

| Fixture | Limitation | Issue that would remove it |
|---|---|---|
| `quoting_embedded_newline.csv` | Multi-line quoted record | none planned |
| `header_none_after_preamble.csv` | The result cannot describe preamble lines without a header, and `_first_row_is_data` checks line 0 only | none planned (#131 replaces the row-0 rule but still tests row 0) |
| `quoting_newline_in_head_window.csv` | Grounding splits lines on every line break, including quoted ones | none planned |
| `quoting_newline_in_tail_window.csv` | Same, and the tail can start mid-record | none planned |
| `footer_longer_than_tail_window.csv` | The footer is longer than the tail window; only its last lines can be anchored (workaround: a larger `tail_bytes`) | none planned (#134 bounds lines on top of the byte windows; it does not widen the tail) |
| `gen_very_wide_long_lines.csv` (matrix) | 200 columns with ~8 KB lines: the default head window holds less than one complete line, so byte windows cannot show a header | #134 (line-based windows) |

#135 (a deterministic pre-pass for encoding and delimiter) removes none of
these.

## Changing the prompt

The prompt is part of every measurement, so a prompt change follows this
checklist in one pull request:

1. Bump `PROMPT_VERSION` in `src/csv_inspector/_prompt.py`. Every
   `result.usage.prompt_version` and every run summary records it, so runs
   of different prompts are never mixed by accident.
2. Run the baseline before the change and the candidate after it, with the
   same flags (see [The ritual](#the-ritual)).
3. Paste the `compare_runs.py` table and verdict changes in the pull
   request.
4. Update the golden prompt strings in `tests/test_prompt_budget.py` and,
   if the template grew, `PROMPT_TEMPLATE_MAX_CHARS` there, deliberately.

## Cloud subset

Fifteen fixtures, one or two per category, none a known limitation, for
reproducible cloud runs that fit a day's free-tier quota. Using the model as
its own fallback makes each fixture's worst case two requests (one plus a
429/503 retry), so `--max-calls 30` lets all fifteen run; a normal run makes
about fifteen requests.

```bash
uv run --directory agents/csv_inspector python scripts/eval_samples.py --backend api \
  --model MODEL --fallback-model MODEL --max-calls 30 --rpm 5 --out "runs/{model}.jsonl" \
  --fixture delimiter_semicolon_decimal_comma.csv \
  --fixture delimiter_tab_commas_quoted_header.tsv \
  --fixture encoding_cp1252_tail_only.csv \
  --fixture encoding_utf16le_bom.csv \
  --fixture header_and_footer_combined.csv \
  --fixture footer_like_data_row_numeric_label.csv \
  --fixture header_none_data_only.csv \
  --fixture quoting_backslash_escape.csv \
  --fixture quoting_doubled_quotes.csv \
  --fixture ragged_rows_inconsistent_columns.csv \
  --fixture single_column.csv \
  --fixture numeric_european_format.csv \
  --fixture null_representations_mixed.csv \
  --fixture gen_combo_eu_legacy.csv \
  --fixture gen_combo_bom_crlf_preamble.csv
```

| Category | Fixtures |
|---|---|
| `delimiter` | `delimiter_semicolon_decimal_comma.csv`, `delimiter_tab_commas_quoted_header.tsv` |
| `encoding` | `encoding_cp1252_tail_only.csv`, `encoding_utf16le_bom.csv` |
| `header_footer` | `header_and_footer_combined.csv`, `footer_like_data_row_numeric_label.csv`, `header_none_data_only.csv` |
| `quoting` | `quoting_backslash_escape.csv`, `quoting_doubled_quotes.csv` |
| `structural` | `ragged_rows_inconsistent_columns.csv`, `single_column.csv` |
| `data_format` | `numeric_european_format.csv`, `null_representations_mixed.csv` |
| `combo` | `gen_combo_eu_legacy.csv`, `gen_combo_bom_crlf_preamble.csv` |

No cloud results are recorded here yet.
