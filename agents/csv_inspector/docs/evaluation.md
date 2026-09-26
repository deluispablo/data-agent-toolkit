# Evaluating csv-inspector against the fixture catalog

`pytest` never measures how accurate the model is: it runs without a model.
Accuracy is measured by `scripts/eval_samples.py`, which runs a real model
against every fixture in `samples/` and scores its answers against the
ground truth in `samples/manifest.json`, and by `scripts/compare_runs.py`,
which puts two or more runs side by side. Both are entry points of the
`scripts/eval_harness/` package (its modules are listed in `ARCHITECTURE.md`)
and live in the repository only; they are not part of the installed package.

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

While iterating, run the [quick subset](#quick-subset) with `--repeat 2`;
the table that goes in the pull request is a full-catalog `--repeat 3`
run. Use the same model, `--bytes`, `--tail-bytes`, `--timeout` and `--repeat`
for both runs; the summary line records all of them, and the table shows
the model and prompt version of each run so mixed comparisons are visible.
Local runs cost nothing but time; for cloud runs read
[Quota notes](#quota-notes) first.

### Replaying a run

A change to grounding, validation or parsing does not change what the
model answers, only what the library makes of it. Measure it without a
model: replay a `--keep-raw` run on the new code.

```bash
uv run --directory agents/csv_inspector python scripts/eval_samples.py   --replay runs/qwen2.5-coder-7b-040.jsonl --out runs/replay.jsonl
uv run --directory agents/csv_inspector python scripts/compare_runs.py   runs/qwen2.5-coder-7b-040.jsonl runs/replay.jsonl
```

Each (fixture, repeat) of the run is inspected again with a
`model_invoker` that answers with the text recorded for it: the
primary's, then the fallback's on the next call, as live. An attempt that
left no text (it timed out) fails again. No model is called, Ollama need
not be running, the whole catalog takes about a second, and the result is
deterministic: replaying the 0.4.0 7b run on the code that recorded it
gives back its 99.7 % and every verdict.

The replay uses the run's models, windows (`--bytes`/`--tail-bytes` may
only repeat them), timeout and repeats; `--category`, `--fixture` and
`--subset` narrow it, and fixtures the run does not hold are skipped and
counted (`replay_skipped`). `--backend`, `--rpm`, `--max-calls` and
`--dry-run` are refused. The summary records `replay_of` and the source
run's `prompt_version`; its prompt tokens are `null` (a custom invoker
reports none) and its latency is not a model's. `compare_runs.py` labels
the column `(replay)` and names the source in its `answers` row. The
replay's lines keep the answers they used, so a replay can be replayed.

**Not valid** for a prompt, schema, sampling or model change: the recorded
answers were given to the old prompt and samples, so replaying them
measures nothing. Those need a live run.

## What is scored

Each fixture's answer is compared field by field with the manifest's
`expected` block: `encoding` (by canonical codec name, alternatives
allowed), `delimiter`, `quotechar`, `escapechar`, `doublequote`,
`has_header`, `header_row_index`, `footer_lines` (surrounding whitespace
ignored), `footer_rows_to_skip` and `columns`. A field without ground truth
is skipped. A fixture whose manifest entry names an `expected_error`
(`empty_file.csv`: `EmptySampleError`, raised before any model call) is
scored on that one pseudo-field instead: raising exactly that exception is
a pass, charged no model call and not counted as an error; returning a
result is a mismatch. A fixture's score is matched / compared; the aggregate score is
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
| `--subset quick\|cloud` | Only the [quick subset](#quick-subset) or the [cloud subset](#cloud-subset) (the lists live in `SUBSETS` in the script). Combines with `--fixture`. |
| `--max-fixtures N` | At most N fixtures: the first N by name, after the filters. |
| `--repeat N` | Run each fixture N times (default 1). |
| `--out PATH` | Write a JSONL run. `PATH` is a `.jsonl` file (one model only), a template containing `{model}` (e.g. `runs/{model}.jsonl`), or a directory that receives `<UTC timestamp>-<model>.jsonl`. An existing file is never overwritten. |
| `--keep-raw` | Store every model answer's raw text in the JSONL lines (`raw_response`). |
| `--rpm N` | Never exceed N model requests per minute (the harness sleeps). |
| `--max-calls N` | Hard stop: never more than N model requests in total, over all models. |
| `--dry-run` | Build the prompts, print each fixture's size and the planned call count, call no model. Needs no credentials. The token estimate assumes 1.81 characters per token (measured on `qwen2.5-coder:7b`). |
| `--replay RUN` | Answer each fixture with the raw answers of a `--keep-raw` run instead of a model (see [Replaying a run](#replaying-a-run)). Refuses `--backend`, `--rpm`, `--max-calls`, `--dry-run` and other windows. |
| `--summarize RUN` | Recover an interrupted run: recompute the summary from the finished lines of `RUN` and append it, marked incomplete. Calls no model; every other flag is ignored. |
| `--env-file`, `--no-env-file`, `--log-level` | As in the CLI. |

`compare_runs.py RUN RUN [RUN ...]`: the first run is the reference; columns
are labelled with the file names, and a replay with `(replay)`.

## Run files

`--out` writes one JSON object per line. The first line is
`{"run": {...}}`: the run's settings (harness and prompt versions, backend,
models, `n_bytes`, `tail_bytes`, timeout, repeat, start time,
`fixtures_planned`). Then comes one line per (fixture, repeat), appended
and flushed as soon as that inspection finishes, so an interrupted run
keeps every finished line:

```json
{"fixture": "delimiter_semicolon.csv", "category": "delimiter", "repeat": 1,
 "known_limitation": false, "model_used": "qwen2.5-coder:7b",
 "matched": ["encoding", "delimiter"], "mismatched": [{"field": "columns", "expected": ["a"], "actual": ["A"]}],
 "skipped": ["escapechar"], "score": 0.9, "columns_recall": 0.0, "columns_count_match": true,
 "usage": {"model": "qwen2.5-coder:7b", "prompt_tokens": 1450, "completion_tokens": 210,
           "latency_seconds": 4.1, "attempts": 1, "retries": 0, "load_seconds": 0.01,
           "prompt_version": "2026.09-a"},
 "latency_seconds": 4.2, "calls": 1, "error": null, "attempt_errors": {}, "retries": 0}
```

`usage` is the inspection's `result.usage` (`null` when it failed; then
`error` and, per model, `attempt_errors` say why). The top-level
`latency_seconds` is measured by the harness around the whole inspection,
failures included. `raw_response` (with `--keep-raw`) is a list of
`{"model", "text"}`, one per model answer in call order. `retries` counts
the transient-error (429/503) retries of the inspection: from `usage` when
it succeeded, and otherwise counted from the library's "retrying once"
warnings, so a cloud fixture that failed after its retry shows it too.
`calls` stays the conservative charge used by `--max-calls` (a failure is
charged its worst case).

The last line is `{"summary": {...}}`: the settings of the `run` line,
whether the run stopped early and whether it is `incomplete`, the
aggregate, per-category and per-field scores, majority-vote scores,
agreement, verdicts, token totals and means, latency p50/p95, model load
time (`load_seconds`: max and p50 over the lines that report one; `null`
on a replay or a cloud run), model calls, fallback uses, retries (every line's `retries`, failed inspections
included), error counts by exception class (the inspection's and each
failed attempt's) and the number of 429 and 503 answers seen in failed
attempts. `fixtures_run` against `fixtures_planned` shows how far the run
got. A live `local` run also records how big its model is once loaded,
asked once from Ollama's `/api/ps` right after the first answer:
`model_size_bytes` and `model_vram_bytes` (the part in GPU memory, `0` on
a CPU-only server); both are `null` when the lookup fails, and replays and
cloud runs have neither key. Files written before 0.7.0 lack these keys;
`compare_runs.py` shows them as `n/a`.

A file with no summary line is an interrupted run (the harness was killed,
or stopped with Ctrl+C, which prints the recovery command).
`compare_runs.py` refuses it. Recover it with:

```bash
uv run --directory agents/csv_inspector python scripts/eval_samples.py --summarize runs/x.jsonl
```

which recomputes the summary from the finished lines, appends it with
`"incomplete": true` and `"stopped_early": "interrupted"`, and prints the
report. `compare_runs.py` then labels its column
`x (incomplete, N/M fixtures)`. Only harness version 2 files (those with a
`run` line) can be recovered.

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
- Use `--subset cloud` to run the [cloud subset](#cloud-subset) instead of
  the whole catalog.

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

## Quick subset

A full-catalog `--repeat 3` run takes about 30 minutes per 7-8B model on
the reference machine. The rule: **iterate on `--subset quick` with
`--repeat 2`, prove on the full catalog with `--repeat 3`** (the table in
the pull request is the full run).

```bash
uv run --directory agents/csv_inspector python scripts/eval_samples.py --no-env-file \
  --model qwen2.5-coder:7b --subset quick --repeat 2 --out "runs/{model}-quick.jsonl"
```

Twenty-one fixtures: every category, every footer kind, header-less files,
a 40-column file, the tab-with-commas files, the blank-name header, the
expected-error file and the misses of the [0.3.0 baseline](#baseline-030).
The list lives in `SUBSETS["quick"]` in `scripts/eval_harness/guards.py`; change
it there and here together.

| Category | Fixtures | Why |
|---|---|---|
| `delimiter` | `delimiter_semicolon_decimal_comma.csv`, `delimiter_tab_commas_quoted_header.tsv` | decimal comma; tab with commas in values |
| `encoding` | `encoding_cp1252_tail_only.csv`, `gen_encoding_utf16le_lf.csv`, `gen_encoding_utf8_lf.csv` | tail-only cp1252; UTF-16 tab with commas (#151); `TOTAL` footer with trailing delimiters (#153) |
| `header_footer` | `gen_footer_none_narrow.csv`, `gen_footer_totals_narrow.csv`, `gen_footer_marker_narrow.csv`, `gen_footer_timestamp_narrow.csv`, `gen_footer_blank_totals_wide.csv`, `footer_end_marker.csv` | every footer kind; a 40-column file (#147); a multi-line footer |
| `header_footer` | `header_none_data_only.csv`, `gen_headerless_marker.csv`, `header_years.csv`, `header_duplicate_and_blank_names.csv`, `header_metadata_banner.csv` | header-less files (#131); a header of years; a blank name (#153); a preamble |
| `quoting` | `quoting_backslash_escape.csv` | escape character |
| `structural` | `single_column.csv`, `empty_file.csv` | one column; the expected `EmptySampleError` |
| `data_format` | `numeric_european_format.csv` | European numbers |
| `combo` | `gen_combo_eu_legacy.csv` | semicolon + decimal comma + cp1252 + preamble + totals |

## Cloud subset

Fifteen fixtures, one or two per category, none a known limitation, for
reproducible cloud runs that fit a day's free-tier quota (about 20 requests
per model per day). Using the model as its own fallback makes each
fixture's worst case two requests (one plus a 429/503 retry). A clean
fixture is charged one request, so with `--max-calls 18` the fifteenth
fixture still starts (14 used plus a worst case of 2), and the spare calls
absorb a retry or a failure; when more fixtures fail, the run stops early
instead of spending the day's quota. Never raise `--max-calls` above 18 on
a free-tier key.

```bash
uv run --directory agents/csv_inspector python scripts/eval_samples.py --backend api \
  --model MODEL --fallback-model MODEL --subset cloud --max-calls 18 --rpm 10 \
  --out "runs/{model}.jsonl"
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

The first cloud results are in [Baseline 0.3.0](#baseline-030) (item 6).

## Line bounds (#134)

The byte windows (4 KiB each) bound memory; `MAX_HEAD_LINES` and
`MAX_TAIL_LINES` in `_sampling.py` bound the lines of each window that
reach the prompt and grounding. On 0.4.0 the samples were about 1,940 of
the 2,639 prompt tokens. The pair was chosen on `qwen2.5-coder:7b` on
2026-09-25: the quick subset for every pair, then the full catalog for the
best ones, keeping **the smallest pair within 0.5 pt of the 0.4.0 accuracy
with no errored inspection**.

Quick subset, `--repeat 2` (21 fixtures; replaying the #158 run on the
same fixtures gives 99.3 % at 3,100 prompt tokens without bounds):

| head x tail | 5 | 10 | 15 |
|---|---|---|---|
| 10 | 98.4 %, 1,064 tok | 100 %, 1,208 tok | 100 %, 1,284 tok |
| 20 | 98.4 %, 1,222 tok | 100 %, 1,366 tok | 100 %, 1,442 tok |
| 30 | 97.4 %, 1,381 tok | 100 %, 1,524 tok | 100 %, 1,600 tok |
| 50 | 98.4 %, 1,664 tok | 100 %, 1,808 tok | 100 %, 1,884 tok |

Five tail lines miss the second line of a multi-line footer
(`footer_end_marker.csv`). Full catalog, `--repeat 3`, against the run of
`main` after #158 (`runs/qwen2.5-coder-7b-m602.jsonl`, no bounds):

| | no bounds | 10 x 10 | 10 x 15 | **15 x 10** |
|---|---|---|---|---|
| prompt tokens (mean) | 2,639 | 1,441 (-45 %) | 1,522 (-42 %) | **1,522 (-42 %)** |
| completion tokens (mean) | 146 | 146 | 146 | 146 |
| latency p50 / p95 | 1.56 s / 6.62 s | 1.49 s / 6.57 s | 1.47 s / 8.33 s | 1.48 s / 8.46 s |
| errored lines | 0 | 3 | 3 | **0** |
| accuracy | 99.8 % | 100.0 % | 100.0 % | **99.7 %** |

Per fixture size (prompt tokens, accuracy):

| size | lines | no bounds | 15 x 10 |
|---|---|---|---|
| < 4 KiB | 135 | 1,071, 100 % | 938, 100 % |
| 4-16 KiB | 87 | 4,759, 99.5 % | 2,114, 99.2 % |
| >= 16 KiB | 18 | 3,888, 100 % | 2,944, 100 % |

Small files gain little (the template dominates); files of 4 KiB and more
lose more than half their prompt. Wide files (40 columns, 200 columns) keep
their byte-bound size: fewer lines than the bounds fit in 4 KiB.

The errored lines of 10 x 10 and 10 x 15 are one fixture,
`gen_preamble_5_footer.csv` (tab-delimited, totals row `TOTAL\t\t…\t\t\t`):
both models copy the footer line and keep repeating `\t` until Ollama
aborts with "token repeat limit reached". Whether it happens depends on
the exact prompt, not on how many lines it holds (with one repeat: 12 x 10
fails, 15 x 10 passes, 15 x 15 fails, 20 x 15 passes); the 0.4.0 prompt
passed it. Telling the model to leave out trailing empty fields made it
worse, so the prompt keeps "verbatim". 15 x 10 passed it on every repeat
(4 of 4); its one new miss is the totals footer of
`gen_encoding_utf8_bom_crlf.csv`, which the model answers as absent on all
3 repeats. The loop is a risk on any tab file whose footer has trailing
empty fields; #138 caps the reply so it fails faster.

## Cost levers

Measured on `qwen2.5-coder:7b` on 2026-09-25, `--repeat 3`, prompt tokens
(mean) and accuracy per category ([#136](https://github.com/deluispablo/data-agent-toolkit/issues/136)):

| category | no line bounds (0.4.0 sampling) | line bounds 15 x 10 | line bounds + `--tail-bytes 0` |
|---|---|---|---|
| `structural` (14 fixtures) | 2,048, 100 % | 1,645, 100 % | 1,170, 96.6 % |
| `delimiter` (8 fixtures) | 801, 100 % | 801, 100 % | 801, 100 % |
| whole catalog | 2,639, 99.8 % | 1,522, 99.7 % | (not run) |

- **Line bounds** (#134, always on): see [Line bounds](#line-bounds-134).
- **`tail_bytes=0`**, the no-footer mode: one read instead of two and no
  tail in the prompt; `footer_lines` is always `[]` and, on a file larger
  than the head window, `covers_whole_file` is `False`. The `delimiter`
  fixtures all fit in the head window, so nothing changes there. On
  `structural` the prompt shrinks by another 29 %; the two misses are
  `gen_large_ledger.csv` and `gen_large_wide.csv`, whose totals footer
  this mode cannot see by design. Use it only when your files never have
  footers.

Run files: `runs/tail0-structural.jsonl`, `runs/tail0-delimiter.jsonl`
(`--category C --tail-bytes 0 --repeat 3 --keep-raw`), against the
full-catalog 15 x 10 run of [Line bounds](#line-bounds-134).

## Comparing models

Choosing a default model is a measurement with its rule written first.
The rule of M8 (milestone "Model selection", 2026-09-26), verbatim:

> Criterion, fixed before any run: the smallest model that scores >= 99.0 % on the full catalog (--repeat 3, no field under 95 %, repeat agreement >= 95 %), ranked by loaded size (ollama ps), then CPU p50 latency, then completion tokens.

Run every candidate as its own fallback (`--model M --fallback-model M`)
so each row measures one model, always with `--no-env-file`, one run at a
time (two runs at once share the machine and skew the latency). Name the
files by phase:

| file | run |
|---|---|
| `runs/{model}-quick.jsonl` | screen: `--subset quick --repeat 1 --keep-raw`, on GPU |
| `runs/{model}-gpu.jsonl` | prove: full catalog, `--repeat 3 --keep-raw`, on GPU |
| `runs/{model}-cpu.jsonl` | time: full catalog, `--repeat 1 --timeout 600`, CPU only |

`compare_runs.py` shows each run's "load max", "loaded size (GB)" and
"in VRAM (GB)" next to its latency.

### CPU-only recipe (Windows)

The cheapest host has no GPU; to time it on a machine that has one, the
Ollama server must not see the GPU:

1. Stop the Ollama tray app, which holds port 11434 (PowerShell:
   `Get-Process ollama* | Stop-Process -Force`).
2. In a terminal, hide the GPU from both of Ollama's GPU backends and
   start the server. Hiding CUDA alone is not enough: Ollama 0.34 then
   finds the same card through Vulkan and loads the model there.

   ```powershell
   $env:CUDA_VISIBLE_DEVICES = '-1'
   $env:GGML_VK_VISIBLE_DEVICES = '-1'
   ollama serve
   ```

3. After the first call of a run, confirm in another terminal that
   `ollama ps` shows `100% CPU` for the model; the run's summary then has
   `model_vram_bytes` 0.
4. When done, stop that server and start the tray app again.

On Linux the same steps apply with `CUDA_VISIBLE_DEVICES=-1
GGML_VK_VISIBLE_DEVICES=-1 ollama serve` after stopping the service.

Knobs worth one run each on CPU, all set in the shell before `ollama serve`
(one server restart per variant):

- `OLLAMA_NUM_PARALLEL=1`: one request slot, so the KV cache is sized for
  one conversation.
- `OLLAMA_KEEP_ALIVE`: how long a model stays loaded after its last call
  (default 5 minutes); longer avoids reloads between sparse inspections.
- `OLLAMA_FLASH_ATTENTION=1`, and with it `OLLAMA_KV_CACHE_TYPE=q8_0`: a
  smaller KV cache.
- Quantisation tags, e.g. `qwen2.5-coder:7b-instruct-q4_0` or `-q8_0`:
  smaller or more faithful weights than the default tag.

### Model comparison 2026-09

The M8 measurement ([#148](https://github.com/deluispablo/data-agent-toolkit/issues/148)),
run on 2026-09-26 with the rule above fixed before the first run.

| | |
|---|---|
| Machine | i5-13600KF (20 threads), 64 GB RAM, RTX 4070 12 GB, Windows 11 |
| Ollama | 0.34.4; GPU runs on the tray app's server, CPU runs on `ollama serve` with CUDA and Vulkan hidden (`ollama ps`: `100% CPU`) |
| `PROMPT_VERSION` | `2026.09-n` (harness version 2), every run |
| Settings | each model as its own fallback, `n_bytes` 4096, `tail_bytes` 4096, `--keep-raw`, `--no-env-file`; timeout 300 s on GPU, 600 s on CPU; loaded models stopped before each run, so "load max" is a cold start |
| Qwen3 | thinking turned off (`think: false`, [#198](https://github.com/deluispablo/data-agent-toolkit/issues/198)) |

**Phases.** (1) Screen: every candidate on the [quick subset](#quick-subset),
`--repeat 1`, GPU; cut below 97 % or with more than one fixture without a
valid answer. (2) Prove: the survivors plus `qwen2.5-coder:3b` on the full
catalog, `--repeat 3`, GPU, where the qualifier is decided. (3) Time: the
qualifier and the two closest cheaper models on CPU, full catalog,
`--repeat 1`; accuracy stays the phase-2 number (CPU answers differ in the
decimals: 3b scored 99.1 % on its CPU run against 98.4 % on GPU). (4) Tune
the top two on CPU.

Every candidate, smallest on disk first within its phase. Full-catalog
rows are `80 x 3` on GPU; screened-out rows are the quick subset (`21 x 1`)
and say so. "Loaded" is `ollama ps`'s size (`model_size_bytes`), on CPU
where measured, else on GPU; "agreement" is the lowest per-field agreement
between repeats. Tokens are means per call.

| model | disk | loaded | GPU p50 / p95 | CPU p50 / p95 | load max (GPU / CPU) | tokens (prompt / completion) | aggregate | weakest field | errored lines | agreement | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `qwen2.5-coder:1.5b` | 986 MB | 1.37 GB | 0.60 s / 1.94 s | 6.48 s / 29.49 s | 1.88 s / 1.04 s | 1,512 / 150 | 97.2 % | `footer_lines` 88.4 % | 4 | 98.8 % (`footer_lines`) | fails: aggregate, field |
| `qwen2.5-coder:3b` | 1.9 GB | 2.41 GB | 0.91 s / 3.82 s | 12.62 s / 56.86 s | 1.67 s / 1.53 s | 1,531 / 170 | 98.4 % | `footer_lines` 93.9 % | 21 | 100 % | fails: aggregate, field |
| `qwen3:4b` | 2.5 GB | 3.87 GB | 1.19 s / 4.02 s | not run | 1.91 s / - | 1,524 / 160 | 97.2 % | `has_header` 75.0 % | 0 | 100 % | fails: aggregate, field |
| **`qwen2.5-coder:7b`** | 4.7 GB | **5.46 GB** | 1.44 s / 4.45 s | **25.07 s / 105.48 s** | 2.40 s / 3.08 s | 1,507 / 141 | **100.0 %** | every field 100 % | 6 | 100 % | **qualifies** |
| `gemma3:1b` (quick) | 815 MB | 0.95 GB | 0.78 s / 1.85 s | - | 0.00 s | 1,593 / 114 | 77.8 % | `header_row_index` 0 % | 19 | - | cut: accuracy, 19 invalid answers |
| `qwen2.5:1.5b` (quick) | 986 MB | 1.36 GB | 0.67 s / 1.88 s | - | 1.33 s | 1,287 / 157 | 94.2 % | `has_header` 66.7 % | 0 | - | cut: accuracy |
| `llama3.2:1b` (quick) | 1.3 GB | 1.72 GB | 0.58 s / 1.71 s | - | 3.09 s | 1,144 / 110 | 85.6 % | `footer_lines` 52.9 % | 0 | - | cut: accuracy |
| `qwen3:1.7b` (quick) | 1.4 GB | 2.24 GB | 0.83 s / 2.45 s | - | 5.63 s | 1,270 / 124 | 91.2 % | `has_header` 50.0 % | 2 | - | cut: accuracy, 2 invalid answers |
| `granite3.3:2b` (quick) | 1.5 GB | 2.36 GB | 0.81 s / 2.31 s | - | 1.37 s | 1,380 / 120 | 89.6 % | `has_header` 33.3 % | 0 | - | cut: accuracy |
| `qwen2.5:3b` (quick) | 1.9 GB | 2.40 GB | 0.91 s / 2.71 s | - | 1.58 s | 1,320 / 110 | 92.8 % | `has_header` 66.7 % | 1 | - | cut: accuracy |
| `llama3.2:3b` (quick) | 2.0 GB | 3.11 GB | 0.83 s / 3.12 s | - | 1.92 s | 1,084 / 116 | 87.6 % | `has_header` 0 % | 8 | - | cut: accuracy, 8 invalid answers |
| `phi4-mini` (quick) | 2.5 GB | 3.70 GB | 1.22 s / 7.19 s | - | 1.86 s | 927 / 106 | 94.8 % | `has_header` 50.0 % | 6 | - | cut: accuracy, 6 invalid answers |
| `gemma3:4b` (quick) | 3.3 GB | 3.03 GB | 1.50 s / 5.20 s | - | 3.61 s | 1,371 / 138 | 93.8 % | `has_header` 0 % | 3 | - | cut: accuracy, 3 invalid answers |
| `llama3.1:8b` (quick) | 4.9 GB | 5.93 GB | 1.75 s / 5.57 s | - | 3.62 s | 1,134 / 123 | 95.0 % | `has_header` 66.7 % | 0 | - | cut: accuracy |
| `qwen3:8b` (quick) | 5.2 GB | 6.30 GB | 1.88 s / 5.46 s | - | 3.63 s | 1,295 / 112 | 96.9 % | `has_header` 66.7 % | 0 | - | cut: accuracy |

On the quick subset the four finalists scored: `qwen2.5-coder:1.5b`
100 % (1 invalid answer), `qwen2.5-coder:3b` 96.6 % (3; kept as the
current fallback, as the plan required), `qwen3:4b` 97.6 % (1),
`qwen2.5-coder:7b` 100 % (0). Every candidate pulled; none was dropped.
The errored lines of `qwen2.5-coder:7b` and most of `:3b`'s are Ollama's
"token repeat limit reached" on the tab-separated totals footers
([#181](https://github.com/deluispablo/data-agent-toolkit/issues/181));
the others are answers that are not JSON or break the schema
(`gemma3:1b` and `llama3.2:3b` mostly set `header_row_index` with
`has_header` false). Prompt tokens differ by tokenizer (Llama, Phi and
Gemma count fewer). The two 8B models loaded at more than 5 GB and lost on
the screen anyway, so neither would win on size.

**Tuning on CPU.** The top two: `qwen2.5-coder:7b`, the only qualifier,
and `qwen2.5-coder:3b`, the closest to the bar and today's fallback. Each
variant is one quick-subset run, `--repeat 2`, against the same run of the
default setup; a variant is kept when p50 improves by 10 % or more, or the
loaded size drops by 20 % or more, with the quick-subset accuracy
unchanged. `OLLAMA_NUM_PARALLEL=1` and `OLLAMA_FLASH_ATTENTION=1` were not
run: Ollama 0.34.4 already defaults to one slot and turns flash attention
on (`flash_attn = auto`) on this CPU, so both are the baseline.

| model | variant | loaded | CPU p50 / p95 | quick accuracy | errored lines | kept |
|---|---|---|---|---|---|---|
| `qwen2.5-coder:7b` | default (`q4_K_M`) | 5.46 GB | 13.91 s / 32.05 s | 100.0 % | 0 | baseline |
| `qwen2.5-coder:7b` | `7b-instruct-q4_0` tag | 5.20 GB | 10.51 s / 28.08 s | 100.0 % | 0 | **yes** (p50 -24 %) |
| `qwen2.5-coder:7b` | `OLLAMA_KV_CACHE_TYPE=q8_0` (with `OLLAMA_FLASH_ATTENTION=1`) | 5.22 GB | 13.76 s / 34.42 s | 100.0 % | 0 | no (p50 -1 %, size -4 %) |
| `qwen2.5-coder:3b` | default (`q4_K_M`) | 2.41 GB | 8.48 s / 19.87 s | 96.7 % | 5 | baseline |
| `qwen2.5-coder:3b` | `3b-instruct-q4_0` tag | 2.31 GB | 6.52 s / 24.43 s | 95.2 % | 7 | no (accuracy drops) |
| `qwen2.5-coder:3b` | `OLLAMA_KV_CACHE_TYPE=q8_0` (with `OLLAMA_FLASH_ATTENTION=1`) | 2.26 GB | 8.28 s / 22.04 s | 96.7 % | 4 | no (p50 -2 %, size -6 %) |

Run files, all in `runs/`: `{model}-quick.jsonl` for every candidate,
`{model}-gpu.jsonl` and `{model}-cpu.jsonl` for the finalists,
`{model}-tune.jsonl` and `{model}-tune-kvq8.jsonl` for the tuning runs
(`{model}` made file-safe, e.g. `qwen2.5-coder-7b-instruct-q4_0-tune.jsonl`),
and `probe-qwen3-4b*.jsonl` from #198.

**Recommendation.** Primary `qwen2.5-coder:7b`, unchanged: it is the only
qualifier (100.0 %, every field and every repeat agreement at 100 %).
Fallback, by the rule: no other model qualifies, so it is the primary
again, `qwen2.5-coder:7b`, which differs from today's `qwen2.5-coder:3b`
(98.4 %, `footer_lines` 93.9 %: under both bars). The library skips a
fallback equal to the primary, so in practice there is no fallback model:
the primary gets the whole time budget instead of about 70 % of it, and a
failed answer raises `InspectionFailedError` instead of asking the 3B
model. On the cheapest host (CPU only) the primary needs about 5.5 GB of
RAM once loaded (5.2 GB with the `7b-instruct-q4_0` tag, the one tuning
variant kept: p50 24 % lower on the quick subset at the same accuracy, not
proven on the full catalog) and answers in 25 s at p50 and 105 s at p95 on
this 20-thread CPU (slowest 121 s, inside the CLI's 300 s budget); a cold
load adds about 3 s. Against today's `qwen2.5-coder:7b` + `:3b` pair
nothing is lost in accuracy; what is lost is the 3B rescue of the #181
lines: 3 errored lines out of 240 with the 3B fallback (the 0.6.0 run)
against 6 without it. Nothing under 5 GB loaded reaches the bar: the best
small models, `qwen2.5-coder:1.5b` (1.4 GB, CPU p50 6.5 s) and `qwen3:4b`,
stop at 97.2 %, so a host that cannot hold 5.5 GB has no default that
meets it.

## Publishing a baseline

A baseline is a full local run (`--repeat 3`) plus the cloud subset,
written up below as `## Baseline <version>`, usually once per release.
The package README quotes it in "Accuracy at a glance", so a pull request
that publishes a new baseline also updates that section in the same PR:
the catalog size, the date and version, the local and cloud accuracy,
latency and cost, and the weakest fields and known failure modes it names.
Everyday before/after runs from [the ritual](#the-ritual) do not touch the
README.

If a demo file's result changes too (a prompt, grounding or default-model
change), regenerate the README's hero, terminal recording and walkthrough: see
"Development" in the README.

## Baseline 0.7.0

The reference after M8 ([#199](https://github.com/deluispablo/data-agent-toolkit/issues/199)):
the local defaults are `qwen2.5-coder:7b` as primary and as fallback,
chosen by the [model comparison](#model-comparison-2026-09), whose
finalist runs of that model are this baseline. Measured on 2026-09-26 on
the same machine as the earlier baselines. The prompt is unchanged since
0.5.0, so the cloud numbers are carried over from
[0.5.0](#baseline-050) (not re-run).

| | |
|---|---|
| Harness version | 2 (summaries now carry load time and loaded size, #197) |
| `PROMPT_VERSION` | `2026.09-n` |
| Catalog | 80 fixtures (44 hand-written, 36 matrix), 6 known limitations |
| GPU run | `runs/qwen2.5-coder-7b-gpu.jsonl`: `qwen2.5-coder:7b` as its own fallback, `n_bytes` 4096, `tail_bytes` 4096, timeout 300 s, `--repeat 3 --keep-raw` (240 inspections), RTX 4070 |
| CPU run | `runs/qwen2.5-coder-7b-cpu.jsonl`: same settings, `--repeat 1`, timeout 600 s, Ollama with CUDA and Vulkan hidden (`ollama ps`: `100% CPU`), i5-13600KF (20 threads) |

Commands, from the repository root:

```bash
uv run --directory agents/csv_inspector python scripts/eval_samples.py --no-env-file \
  --model qwen2.5-coder:7b --fallback-model qwen2.5-coder:7b --repeat 3 --keep-raw \
  --out "runs/{model}-gpu.jsonl"
# CPU: restart Ollama as in "CPU-only recipe", then
uv run --directory agents/csv_inspector python scripts/eval_samples.py --no-env-file \
  --model qwen2.5-coder:7b --fallback-model qwen2.5-coder:7b --repeat 1 --timeout 600 \
  --keep-raw --out "runs/{model}-cpu.jsonl"
```

| metric | 7b, 0.5.0 (fallback 3b) | **7b, 0.7.0, GPU** | **7b, 0.7.0, CPU** |
|---|---|---|---|
| fixtures x repeat | 80 x 3 | 80 x 3 | 80 x 1 |
| prompt tokens (mean) | 1,522 | 1,507 | 1,507 |
| completion tokens (mean) | 144 | 141 | 141 |
| latency p50 | 1.49 s | **1.44 s** | **25.07 s** |
| latency p95 | 4.69 s | **4.45 s** | **105.48 s** (slowest 120.9 s) |
| load max (cold) | not recorded | 2.40 s | 3.08 s |
| loaded size | not recorded | 5.13 GB, all in VRAM | 5.46 GB, 0 in VRAM |
| errored lines | 3 | 6 | 2 |
| **accuracy** | **100.0 %** | **100.0 %** | 99.7 % |
| majority-vote accuracy | 100.0 % | 100.0 % | n/a (one repeat) |

Every category and every field scores 100 % on the GPU run, and every
field agrees across the three repeats. The CPU pass differs in the
decimals (it misses the footer of `gen_encoding_utf8_bom_crlf.csv`:
`footer_lines` and `footer_rows_to_skip` at 98.1 %); the accuracy of
record is the GPU run's. The errored lines are all
[#181](https://github.com/deluispablo/data-agent-toolkit/issues/181):
`gen_preamble_5_footer.csv` and `gen_utf16le_bom_crlf_wide.csv`, both
tab-separated with a totals row that ends in empty fields, make the model
repeat `\t` until Ollama aborts, on every repeat. In 0.5.0 and 0.6.0 the
`qwen2.5-coder:3b` fallback answered the second one; with the fallback
now the primary itself, both fail. Cloud: as measured at 0.5.0.

## Baseline 0.5.0

The reference for M7 ([#140](https://github.com/deluispablo/data-agent-toolkit/issues/140)).
Measured on 2026-09-25 with the library of `main` after M6 (the M6-05 runs;
`main` differs from that branch only in the API example and docs), on the
same machine as the earlier baselines.

| | |
|---|---|
| Harness version | 2 |
| `PROMPT_VERSION` | `2026.09-n` |
| Catalog | 80 fixtures (44 hand-written, 36 matrix), 6 known limitations; `escapechar`/`doublequote` scored on 45 (#158) |
| Local runs | `runs/qwen2.5-coder-7b-050.jsonl` and `runs/qwen2.5-coder-3b-050.jsonl`: each model as primary, default fallback `qwen2.5-coder:3b`, `n_bytes` 4096, `tail_bytes` 4096, timeout 300 s, `--repeat 3 --keep-raw` (240 inspections each) |
| Cloud run | `runs/gemini-flash-lite-latest-050.jsonl`: `gemini-flash-lite-latest` as its own fallback, `--subset cloud --max-calls 18 --rpm 10` |

Commands: as for [0.4.0](#baseline-040), with `-050` in the file names.

### Headline, 0.3.0 against 0.4.0 and 0.5.0

| metric | 7b, 0.3.0 | 7b, 0.4.0 | **7b, 0.5.0** | 3b, 0.5.0 | cloud, 0.4.0 | **cloud, 0.5.0** |
|---|---|---|---|---|---|---|
| prompt version | 2026.09-a | 2026.09-m | 2026.09-n | 2026.09-n | 2026.09-m | 2026.09-n |
| fixtures x repeat | 80 x 3 | 80 x 3 | 80 x 3 | 80 x 3 | 15 x 1 | 15 x 1 |
| prompt tokens (mean) | 2,834 | 2,639 | **1,522** (-42 %) | 1,531 | 1,806 | **968** (-46 %) |
| completion tokens (mean) | 389 | 146 | **144** | 170 | 119 | **114** |
| latency p50 | 5.04 s | 1.58 s | **1.49 s** | 0.92 s | 1.12 s | 19.77 s |
| latency p95 | 20.89 s | 6.52 s | **4.69 s** | 3.75 s | 1.39 s | 28.32 s |
| model calls | 282 | 243 | 243 | 237 | 15 | 15 |
| errored lines | 27 | 0 | **3** | 21 | 0 | 0 |
| **accuracy** | **92.8 %** | **99.7 %** | **100.0 %** | **98.4 %** | **100.0 %** | **100.0 %** |
| majority-vote accuracy | 92.8 % | 99.6 % | 100.0 % | 98.4 % | 100.0 % | 100.0 % |

Accuracy excludes errored lines. The cloud latency is the service's on the
day measured: every call was a single attempt with no retry.

| category | 7b, 0.4.0 | 7b, 0.5.0 | 3b, 0.5.0 |
|---|---|---|---|
| `combo`, `data_format`, `delimiter`, `encoding`, `quoting`, `structural` | 100 % | 100 % | 100 % |
| `header_footer` | 99.2 % | 100 % | 94.9 % |

| field | 7b, 0.4.0 | 7b, 0.5.0 | 3b, 0.5.0 |
|---|---|---|---|
| `encoding`, `delimiter`, `quotechar`, `escapechar`, `doublequote`, `has_header` | 100 % | 100 % | 100 % |
| `header_row_index` | 100 % | 100 % | 98.4 % |
| `footer_lines`, `footer_rows_to_skip` | 98.8 % | 100 % | 93.9 % |
| `columns` (exact list) | 100 % | 100 % | 98.5 % |
| `expected_error` | 100 % | 100 % | 100 % |

### What moved the numbers

| Change | Issue | Effect |
|---|---|---|
| Harness `--replay` | #166 | grounding changes measured in a second, no model |
| `escapechar=None` for quoted fields without escaped quotes; quote escaping scored on 45 fixtures | #158 | no verdict change (0.4.0 already answered `null`); aggregate +0.1 pt from more compared fields |
| Samples bounded to 15 head and 10 tail lines | #134 | prompt tokens -42 % (4-16 KiB files: 4,759 -> 2,114); see [Line bounds](#line-bounds-134) |
| Reply cap from the head's field count, three `num_ctx` windows | #138 | latency p95 6.71 s -> 4.69 s (7b), 7.06 s -> 3.75 s (3b): looping answers stop at ~450 tokens |

### Failure modes

- **7b**: `gen_preamble_5_footer.csv` errors on all 3 repeats: both models
  repeat `\t` inside `footer_first_line` (its totals row ends in empty tab
  fields) until Ollama aborts with "token repeat limit reached". It passed
  on 0.4.0 and depends on tiny prompt and numeric differences
  ([#181](https://github.com/deluispablo/data-agent-toolkit/issues/181)).
  The other misses are the known limitations.
- **3b as primary**: 21 errored lines, mostly the looping answers of 0.4.0
  (now cut at the reply cap) plus the tab-footer loop; footers are the
  weakest field (93.9 %). With the default configuration 3b answers only
  after a 7b failure.
- No 429 or 503 on the cloud run; no retries anywhere.

### Cloud cost

`gemini-flash-lite-latest` on the cloud subset: 14,514 prompt and 1,710
completion tokens for 15 inspections, about **$0.0086** at the list price
used before ($0.30 input and $2.50 output per million tokens): **$0.00058
per inspection, about $0.58 per 1,000 files** (0.4.0: $0.84).

## Baseline 0.4.0

The reference every M6 pull request compares against
([#139](https://github.com/deluispablo/data-agent-toolkit/issues/139)).
Measured on 2026-09-25 on `main` after M5, with the same machine as the
[0.3.0 baseline](#baseline-030).

| | |
|---|---|
| Harness version | 2 |
| `PROMPT_VERSION` | `2026.09-m` |
| Catalog | 80 fixtures (44 hand-written, 36 matrix), 6 known limitations |
| Local runs | `runs/qwen2.5-coder-7b-040.jsonl` and `runs/qwen2.5-coder-3b-040.jsonl`: each model as primary, default fallback `qwen2.5-coder:3b`, `n_bytes` 4096, `tail_bytes` 4096, timeout 300 s, `--repeat 3 --keep-raw` (240 inspections each) |
| Cloud run | `runs/gemini-flash-lite-latest-040.jsonl`: `gemini-flash-lite-latest` as its own fallback, `--subset cloud --max-calls 18 --rpm 10` |

Commands (from `agents/csv_inspector`; run files are git-ignored):

```bash
uv run python scripts/eval_samples.py --no-env-file --model qwen2.5-coder:7b \
  --repeat 3 --keep-raw --out "runs/{model}-040.jsonl"
uv run python scripts/eval_samples.py --no-env-file --model qwen2.5-coder:3b \
  --repeat 3 --keep-raw --out "runs/{model}-040.jsonl"
uv run python scripts/eval_samples.py --backend api --env-file .env \
  --model gemini-flash-lite-latest --fallback-model gemini-flash-lite-latest \
  --subset cloud --max-calls 18 --rpm 10 --keep-raw --out "runs/{model}-040.jsonl"
```

### Headline, 0.3.0 against 0.4.0

| metric | 7b, 0.3.0 | **7b, 0.4.0** | 3b, 0.4.0 | cloud, 0.3.0 | **cloud, 0.4.0** |
|---|---|---|---|---|---|
| prompt version | 2026.09-a | 2026.09-m | 2026.09-m | 2026.09-a | 2026.09-m |
| fixtures x repeat | 80 x 3 | 80 x 3 | 80 x 3 | 15 x 1 | 15 x 1 |
| prompt tokens (mean) | 2,834 | **2,639** (-7 %) | 2,616 | 2,118 | **1,806** (-15 %) |
| completion tokens (mean) | 389 | **146** (-62 %) | 164 | 403 | **119** (-70 %) |
| latency p50 | 5.04 s | **1.58 s** | 1.02 s | 1.75 s | 1.12 s |
| latency p95 | 20.89 s | **6.52 s** | 7.51 s | 2.48 s | 1.39 s |
| model calls | 282 | 243 | 237 | 15 | 15 |
| errored lines | 27 | **0** | 14 | 0 | 0 |
| **accuracy** | **92.8 %** | **99.7 %** | **98.7 %** | **98.1 %** | **100.0 %** |
| majority-vote accuracy | 92.8 % | 99.6 % | 98.7 % | 98.1 % | 100.0 % |

| category | 7b, 0.3.0 | 7b, 0.4.0 | 3b, 0.4.0 |
|---|---|---|---|
| `combo`, `data_format`, `delimiter` | 100 % | 100 % | 100 % |
| `encoding` | 85.7 % | 100 % | 100 % |
| `header_footer` | 87.9 % | 99.2 % | 96.0 % |
| `quoting` | 97.6 % | 100 % | 100 % |
| `structural` | 95.7 % | 100 % | 100 % |

| field | 7b, 0.3.0 | 7b, 0.4.0 | 3b, 0.4.0 |
|---|---|---|---|
| `encoding`, `quotechar`, `escapechar` | 100 % | 100 % | 100 % |
| `delimiter` | 92.3 % | 100 % | 100 % |
| `doublequote` | 50.0 % | 100 % | 100 % |
| `has_header` | 75.0 % | 100 % | 100 % |
| `header_row_index` | 96.8 % | 100 % | 98.5 % |
| `footer_lines`, `footer_rows_to_skip` | 77.1 % | 98.8 % | 96.1 % |
| `columns` (exact list) | 96.9 % | 100 % | 98.5 % |
| `expected_error` | (not scored) | 100 % | 100 % |

### What moved the numbers

| Change | Issue | Effect in the M5 runs |
|---|---|---|
| Grounding: header-less shape test, delimiter dominance, footer anchors | #131, #151, #153 | 7b accuracy 92.8 % -> 98.3 %; `footer_lines` 77.1 % -> 96.5 % |
| Columns as names only; `notes`, types and examples dropped | #129, #147 | completion 389 -> 172 tokens; p50 5.2 s -> 1.65 s; the eight 40-column files no longer hit the reply cap |
| Response schema on both backends, first footer line only, `_ModelAnswer` | #130, #132 | 0 errored lines on 7b; accuracy 100.0 % on the post-wave-4 run |
| Prompt rewrite | #133 | template 852 -> 701 tokens (see [Prompt passes](#prompt-passes-133)) |

### Failure modes

- **7b**: no errored inspection. Three verdicts fail: two known
  limitations (`footer_longer_than_tail_window.csv`,
  `gen_very_wide_long_lines.csv`) and `gen_headerless_marker.csv`, whose
  footer the model misses.
- **3b as primary**: 14 errored lines (17 exceptions, known limitations
  included) on 4 regular files (`single_column.csv`, `header_years.csv`,
  `gen_combo_pipe_ragged.csv`, `gen_footer_none_narrow.csv`). The model
  repeats the prompt's footer examples inside `columns` until the
  1,024-token reply cap, so the answer is not valid JSON. The fallback is
  the same model in this run; with the default configuration (7b primary)
  3b only answers after a 7b failure, which the 7b run above never had.
  Reply sizing is #138.
- No 429 or 503 on the cloud run; no retries anywhere.

### Cloud cost

`gemini-flash-lite-latest` on the cloud subset: 27,083 prompt and 1,786
completion tokens for 15 inspections, which costs about **$0.0126** at
the list price used for 0.3.0 ($0.30 input and $2.50 output per million
tokens). That is **$0.00084 per inspection, about $0.84 per 1,000 files**:
half of 0.3.0's $1.64, mostly because the answer is 70 % shorter.

### Correction: quote escaping scored on every quoted fixture (#158)

The baselines above scored `escapechar` on 1 fixture and `doublequote` on
2, so a model that always answered `"\\"` scored 100 %. Since #158 the
manifest holds both on the 45 fixtures with quoted fields (44 expect
`escapechar: null`), and an expected `null` is scored. Replaying the 0.4.0
answers under the new scoring (`--replay`, same answers, no model):

| | 7b, 0.4.0 | 7b, rescored | 3b, 0.4.0 | 3b, rescored |
|---|---|---|---|---|
| accuracy | 99.7 % | 99.8 % | 98.7 % | 98.8 % |
| `escapechar`, `doublequote` | 100 % (1 and 2 fixtures) | 100 % (45 fixtures) | 100 % | 100 % |

No verdict changed: the 0.4.0 prompt already answered `null` on these
files, and the doubled quote in the matrix fixtures' `Taller "El Rápido"`
was grounded since #130. The aggregate moves only because more fields are
compared. The grounding rule #158 adds (quoted fields with no escaped
quote mean `escapechar=None`) guards the answer the model gave on 0.3.0
and on the README's demo files.

### Open for M6

- The template is 701 tokens, not the 600 that #133 targeted: the footer
  rules proved necessary (see [Prompt passes](#prompt-passes-133)).
- The 3b loops above, and `num_predict` sizing (#138).
- The known limitations are unchanged, except that
  `header_none_after_preamble.csv` now passes on 7b.

### Prompt passes (#133)

The 0.4.0 prompt was cut pass by pass from `2026.09-f` (after #129, #130,
#132), one commit per pass. Characters are the instruction template with
a tail section and empty samples (`build_prompt("", "utf-8",
tail_sample="")`). Tokens are that template's `prompt_tokens` on
`qwen2.5-coder:7b`, system prompt and chat wrapping included (the command
in #133). The quick subset ran on `qwen2.5-coder:7b` with `--repeat 2`;
"errored" counts inspections that failed validation.

| Pass | `PROMPT_VERSION` | Change | Characters | Tokens | Quick subset |
|---|---|---|---|---|---|
| 0 | `2026.09-f` | after #129, #130, #132 | 3,274 | 852 | 100.0 %, 0 errored |
| A | `2026.09-g` | "Keep in mind" list dropped; delimiter note kept | 2,980 | 791 | 100.0 %, 2 errored (`single_column.csv`: delimiter `"\n"`) |
| B | `2026.09-h` | header and footer rules as bullets, one example per footer kind | 2,314 | 638 | 99.2 %, 4 errored (+ `header_none_data_only.csv`: `has_header` true with a null index) |
| C | `2026.09-i` | role sentence dropped | 2,247 | 618 | 96.7 %, 4 errored (a multi-line footer anchor) |
| D | `2026.09-j` | one-line encoding hint, shorter tail note | 2,082 | 586 | **90.2 %**: without "read footer lines ONLY from its last lines" the model answers no footer on six files |
| D2 | `2026.09-k` | D with that clause restored | 2,126 | **595** | 97.4 %, 0 errored; **100.0 %** once its answers are re-grounded with the final grounding |

| L | `2026.09-l` | D2 with the field notes moved after the rules (3b repeated the footer examples into `columns`) | 2,126 | 595 | full catalog: 7b 98.8 %, `combo` -9.5 pts: **blocked** |
| M (kept) | `2026.09-m` | L with the pass-0 FOOTER paragraph restored, all its examples included | 2,567 | **701** | see the full-catalog table below |

Full catalog, `--repeat 3`, pass 0 (`2026.09-f`) against the kept prompt
(`2026.09-m`). Every run's recorded answers are re-grounded with the final
code (`--keep-raw` answers replayed through a custom `model_invoker`), so
the columns differ by prompt only. The 3b pass-0 run had 219 of 240 lines
error on `confidence: 100` before the percentage rule; re-grounded, none
do.

| metric | 7b pass 0 | 7b kept | 3b pass 0 | 3b kept |
|---|---|---|---|---|
| template tokens | 852 | 701 | 852 | 701 |
| prompt tokens (mean) | 2,781 | 2,639 | n/a | 2,616 |
| completion tokens (mean) | 144 | 146 | n/a | 164 |
| latency p50 | 1.51 s | 1.53 s | 0.83 s | 1.00 s |
| errored lines | 0 | 0 | 0 | 14 |
| **accuracy** | **100.0 %** | **99.7 %** | **97.5 %** | **98.7 %** |
| `header_footer` | 100.0 % | 99.2 % | 94.7 % | 96.0 % |
| `structural` | 100.0 % | 100.0 % | 100.0 % | 100.0 % |
| other categories | 100.0 % | 100.0 % | 95.6-100 % | 100.0 % |

What the passes taught:

- **The footer rules carry weight.** Compressing them (pass B) and the
  tail note (pass D) cost footer recall on the full catalog. Pass D's
  clause came back in D2, and the whole pass-0 FOOTER paragraph came back
  in M. This is why the template ends at 701 tokens and not at the
  600-token target: the 150 tokens between them are those rules.
- **Most misses were small-model slips, not the prompt's wording.** The
  prompt had been papering over them: a line break as delimiter,
  `has_header` true with a null index, a multi-line footer anchor, a `'`
  quote character that never occurs, invented names for a header-less
  file, a one-column file listed line by line, and a 3b confidence of
  `100`. Validation and grounding now read them instead.
- **3b still loops on 14 lines** (4 files x up to 3 repeats), repeating
  the prompt's footer examples inside `columns` until the reply cap. The
  answer is then invalid JSON, so the inspection fails over to the
  fallback, which is the same model in this run. The schema caps
  `footer_first_line` at 300 characters, but a list of names cannot be
  capped without a per-file bound; #138 re-sizes the reply cap.

## Baseline 0.3.0

The reference every M5 pull request compared against (M6 compares against [Baseline 0.4.0](#baseline-040))
([#128](https://github.com/deluispablo/data-agent-toolkit/issues/128)).
Measured on 2026-09-24 on `main` after M4. The inspection behaviour is
that of 0.3.0: M4 added measurement only (`Usage`, `PROMPT_VERSION`), and
the prompt text is unchanged.

| | |
|---|---|
| Harness version | 1 |
| `PROMPT_VERSION` | `2026.09-a` |
| Catalog | 80 fixtures (44 hand-written, 36 matrix), 6 known limitations |
| Local run | `runs/qwen2.5-coder-7b-r3.jsonl`: `qwen2.5-coder:7b`, default fallback `qwen2.5-coder:3b`, `n_bytes` 4096, `tail_bytes` 4096, timeout 300 s, `--repeat 3` (240 inspections, 282 model calls, 29 min) |
| Cloud run | `runs/gemini-flash-lite-latest-cloud.jsonl`: `gemini-flash-lite-latest` as its own fallback, [cloud subset](#cloud-subset), 15 calls |
| Machine | Intel i5-13600KF, 64 GB RAM, NVIDIA RTX 4070 12 GB, Windows 11, Ollama 0.34.3 |

Commands (from `agents/csv_inspector`; run files are git-ignored):

```bash
uv run python scripts/eval_samples.py --no-env-file --model qwen2.5-coder:7b \
  --repeat 3 --keep-raw --out "runs/{model}-r3.jsonl"
uv run python scripts/eval_samples.py --backend api --env-file .env \
  --model gemini-flash-lite-latest --fallback-model gemini-flash-lite-latest \
  --max-calls 18 --rpm 10 --keep-raw --out "runs/{model}-cloud.jsonl" \
  <the 15 --fixture flags of the cloud subset; today: --subset cloud>
```

The breakdowns below are arithmetic on those run files (raw answers,
`usage` and fixture sizes); each table states its method.

### 1. Prompt tokens

Mean prompt of a single-attempt call: **2,658 tokens** (198 calls; the
mean over every call, fallbacks included, is 2,834). Ollama has no
tokenize endpoint, so the parts are measured and estimated like this. The
instruction template, system prompt and chat wrapping were measured once on
`qwen2.5-coder:7b` with empty samples: 926 tokens without the tail section,
998 with it, of which the system prompt and chat wrapping are 27. The rest
of each call's `prompt_tokens` is sample text, split between head and tail
by character count at the calibrated 1.81 characters per token. CSV text
tokenizes densely, well below the 2 characters per token that the
library's `num_ctx` sizing assumes.

| Part | Tokens (mean) | Share |
|---|---|---|
| Instruction template + system prompt + chat wrapping | ~950 | 36 % |
| Head sample (1,848 characters mean) | ~1,019 | 38 % |
| Tail sample (1,251 characters mean; none when the head covers the file) | ~689 | 26 % |

Inside the template (3,540 characters), the JSON shape description takes
34 %, the footer rules 27 %, the header rules 14 %, the "Keep in mind"
list 12 % and the introduction and sample markers 12 %.

Cloud: `gemini-flash-lite-latest` used 2,118 prompt tokens per call on the
cloud subset (another tokenizer and a smaller subset: not comparable token
for token).

### 2. Completion tokens

Mean **389 completion tokens** per inspection (p95 589). Shares of the
answer characters over the 210 answers that parse (66 answers were cut off
by the 1,024-token reply cap, see item 5), counting each field's key and
serialized value. Converting characters to tokens proportionally is an
approximation.

| Output field | Share of answer characters |
|---|---|
| `columns[].example_values` | 26.7 % |
| `columns[].inferred_type` + `nullable` | 17.1 % |
| `columns[].name` | 7.1 % |
| `notes` | 7.7 % |
| `footer_lines` | 3.4 % |
| Everything else (dialect fields, `confidence`, JSON punctuation and indentation) | 38.1 % |

Whitespace (indentation and line breaks) is 23.5 % of the answer
characters; it tokenizes cheaply, so its token share is lower.

### 3. Accuracy

Aggregate **92.8 %** over 195 scored inspections (known limitations and
errored inspections excluded). The majority vote over the 3 repeats is also
92.8 %: at `temperature=0` every field agreed in 100 % of the repeats, and
no fixture changed its answer between repeats (no unstable fixture).

| Category | Score |
|---|---|
| `combo`, `data_format`, `delimiter` | 100 % |
| `quoting` | 97.6 % |
| `structural` | 95.7 % |
| `header_footer` | 87.9 % |
| `encoding` | 85.7 % |

| Field | Score |
|---|---|
| `encoding`, `quotechar`, `escapechar` | 100 % |
| `columns` (exact list) | 96.9 % |
| `header_row_index` | 96.8 % |
| `delimiter` | 92.3 % |
| `footer_lines`, `footer_rows_to_skip` | 77.1 % |
| `has_header` | 75.0 % |
| `doublequote` | 50.0 % (2 scored fixtures) |

`columns` diagnostics: mean per-name recall 98.1 %, column count right in
98.5 %. The misses, grouped (each fixture gave the same answer in every
repeat):

- Footer, 9 fixtures: a data row reported as a footer line
  (`encoding_cp1252_tail_only.csv`, `exactly_head_window_size.csv`,
  `footer_after_total_energies_row.csv`, `gen_encoding_cp1252_lf.csv`,
  `gen_encoding_utf8_bom_crlf.csv`, `gen_footer_none_narrow.csv`,
  `gen_headerless_marker.csv`), or a footer missed
  (`footer_end_marker.csv`, `gen_encoding_utf8_lf.csv`).
- Delimiter, 4 tab files reported as `,` because their values hold commas
  and grounding keeps the dominated answer
  ([#151](https://github.com/deluispablo/data-agent-toolkit/issues/151)):
  `gen_encoding_utf16le_lf.csv`, `gen_encoding_utf16le_crlf.csv`,
  `gen_footer_marker_narrow.csv` and `gen_preamble_5_footer.csv` (which also
  gets `header_row_index` 4 for 5). `single_column.csv` is reported as tab.
- `header_years.csv`: a header of years read as a header-less file.
- `header_duplicate_and_blank_names.csv`: the blank first name is dropped.
- `quoting_backslash_escape.csv`: `doublequote` true for a
  backslash-escaped file.

### 4. Latency and reloads

Harness wall time per inspection, all attempts included: p50 **5.0 s**,
p95 **20.9 s**.

| Fixture size | Inspections | p50 | p95 |
|---|---|---|---|
| < 4 KiB (the head covers the file) | 132 | 3.5 s | 6.4 s |
| 4–16 KiB | 72 | 6.0 s | 15.9 s |
| 16–64 KiB | 0 (all fail, item 5) | — | — |
| >= 64 KiB | 3 | 5.6 s | 5.6 s |

Every call reports a `load_seconds` above 0 (a few milliseconds when the
model is resident). Real reloads, over 1 s: **10** of 207 successful
inspections (2.7 to 6.6 s each, about 50 s of the 29-minute run), all in the
first of the three passes; 2 of them are inspections that used the
fallback.

### 5. Failure modes

| Outcome | Count |
|---|---|
| Failed inspections (`InspectionFailedError`) | 30 (27 regular, 3 on a known limitation) |
| of which every attempt was cut off by the 1,024-token reply cap (40-column files) | 24 ([#147](https://github.com/deluispablo/data-agent-toolkit/issues/147)) |
| `EmptySampleError` on `empty_file.csv` (expected, but counted as an error by harness version 1; scored as a pass since [#149](https://github.com/deluispablo/data-agent-toolkit/issues/149)) | 3 |
| Failed attempts with `ResponseParsingError` | 60 |
| `SchemaValidationError`, `InspectionTimeoutError` | 0 |
| Fallback used successfully | 9 |

Every `ResponseParsingError` in the run is a truncated answer, not
malformed JSON. The eight 40-column fixtures fail in every repeat on both
default models, and a 12-column file already exceeds the cap on the 7b
model (it succeeds through the 3b fallback).

### 6. Cloud

`gemini-flash-lite-latest` on the cloud subset, `--max-calls 18 --rpm 10`:
**15 calls, 0 retries, no 429 or 503**, 98.1 % (only
`quoting_backslash_escape.csv` misses, on `escapechar` and `doublequote`),
31,774 prompt and 6,043 completion tokens, p50 1.7 s, p95 2.5 s. At list
price the run costs about **$0.025**: $0.0016 per inspection, about $1.64
per 1,000 files. That price takes the alias as Gemini 3.5 Flash-Lite, the
newest Flash-Lite on Google's pricing page ($0.30 input and $2.50 output
per million tokens, standard paid tier, page dated 2026-09-24). The free
tier costs nothing but allows about 20 requests per model per day.

`gemini-3.6-flash` (the default cloud primary): no result. The key's daily
free-tier quota for it was already spent (8 answers `429
RESOURCE_EXHAUSTED ... limit: 20`, 1 answer 503), and the run stopped at its
18-call limit. It is pending, with the model comparison, in
[#148](https://github.com/deluispablo/data-agent-toolkit/issues/148).

### 7. Decision table for M5/M6

GO: do it as specified. RESCOPE: do it with the issue body adjusted to
these numbers (the issue says what changed). DROP: close. Each issue has a
comment with its row.

| Issue | Baseline number it attacks | Projected effect | Verdict |
|---|---|---|---|
| [#129](https://github.com/deluispablo/data-agent-toolkit/issues/129) columns as names only | `example_values` 26.7 %, `inferred_type` + `nullable` 17.1 %, `notes` 7.7 % of answer characters; 24 inspections cut off (#147) | about half of the 389 mean completion tokens; unblocks the 40-column files | GO |
| [#130](https://github.com/deluispablo/data-agent-toolkit/issues/130) Ollama schema output | JSON shape description = 34 % of the template (~315 tokens) | ~12 % of the mean prompt; no parsing gain measured (every parse failure was a truncation) | GO |
| [#131](https://github.com/deluispablo/data-agent-toolkit/issues/131) row-0 shape rule | `has_header` 75.0 %, `header_footer` 87.9 % | accuracy guard for #129; `header_years.csv` must keep its header | GO |
| [#132](https://github.com/deluispablo/data-agent-toolkit/issues/132) first footer line only | `footer_lines` 77.1 % (weakest dialect field); 3.4 % of answer characters | small completion saving; the value is the `_ModelAnswer` split and footer accuracy | GO |
| [#133](https://github.com/deluispablo/data-agent-toolkit/issues/133) prompt <= 600 tokens | template ~950 tokens measured (not ~1,350), 36 % of the prompt | ~350 tokens per call, ~13 % of the mean prompt | RESCOPE |
| [#134](https://github.com/deluispablo/data-agent-toolkit/issues/134) line windows | samples = 64 % of the prompt (head ~1,019, tail ~689 tokens) | the largest prompt lever on narrow files; more rows on wide ones | GO |
| [#135](https://github.com/deluispablo/data-agent-toolkit/issues/135) deterministic pre-pass | `encoding` already 100 %, `delimiter` 92.3 % (every miss is #151); the two fields are a few output tokens | small token saving; the accuracy part is mostly covered by #151 | RESCOPE |
| [#136](https://github.com/deluispablo/data-agent-toolkit/issues/136) `tail_bytes=0` mode | tail = 26 % of the prompt (~689 tokens) when present | ~26 % of the prompt for hosts that need no footer | GO |
| [#137](https://github.com/deluispablo/data-agent-toolkit/issues/137) quota-aware calls | no per-minute 429 at 10 rpm; all 8 429s were the daily quota; harness guards already shipped in #122 | a per-minute limiter does not address the 429s observed | RESCOPE |
| [#138](https://github.com/deluispablo/data-agent-toolkit/issues/138) num_ctx / num_predict | 10 reloads in 207 inspections (~50 s of 29 min); `num_predict` 1,024 too small for wide files (#147) | little latency to win from reloads; reply sizing is the real fix | RESCOPE |

Issues opened from this baseline: #147 (wide files cut off, M5), #148
(model comparison, deferred from #126), #149 (harness follow-ups) and #151
(delimiter grounding).
