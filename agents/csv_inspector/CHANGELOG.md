# Changelog

All notable changes to `csv-inspector` are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project adheres to [Semantic Versioning](https://semver.org/). While the
version is `0.x`, minor releases may include breaking changes; each one is
listed under **Changed (breaking)**.

## [Unreleased]

To be released as **0.4.0**: the output contract is slimmed to what
pipelines consume.

### Changed (breaking)

- `CSVInspectionResult.columns` is a list of column names (`list[str]`):
  the names as written in the header row, in file order, or `column_1`,
  `column_2`, ... for a header-less file. Surrounding whitespace in a
  model's answer is stripped; empty names (a pandas index column) and
  duplicate names are kept, as they are in the file; an empty list fails
  validation. The per-column `inferred_type`, `nullable` and
  `example_values` are removed, and so is `notes`. `ColumnSchema` and
  `ColumnType` are no longer exported. `confidence` stays: route low values
  to human review. Those fields were 51 % of the characters of a model's
  answer on the 0.3.0 baseline, and the model no longer spends tokens on
  them. The prompt asks for the names only (`PROMPT_VERSION` `2026.09-b`).
  Migration: `[c.name for c in result.columns]` becomes `result.columns`;
  drop any use of `notes`; for column types, let the engine that loads the
  file infer them (pandas by default, Spark `inferSchema`, BigQuery
  schema auto-detection), which reads every row instead of a 4 KB sample:
  see `docs/using-the-result.md`, "Column types"
  ([#129](https://github.com/deluispablo/data-agent-toolkit/issues/129)).

### Added

- `Usage`, the type of `result.usage`, is exported from `csv_inspector`,
  so host code can annotate it
  ([#129](https://github.com/deluispablo/data-agent-toolkit/issues/129)).
- Every successful inspection records what its model phase cost in
  `result.usage`: the model whose answer was kept, prompt and completion
  tokens summed over every attempt (a failed primary's tokens included;
  `None` when not reported, e.g. by a custom `model_invoker`), wall-time
  latency, attempts, transient cloud retries, and Ollama's model load time.
  The built-in invokers read the counts the Ollama and Gemini SDKs already
  return, with no extra call. `usage` is left out of `model_dump()`,
  `model_dump_json()` and `model_json_schema()`, so the JSON contract and
  the schema sent to Gemini are unchanged; it does take part in `==`. One
  INFO log line per success reports it, and the CLI's new `--stats` flag
  prints it to stderr as JSON
  ([#121](https://github.com/deluispablo/data-agent-toolkit/issues/121)).
- The evaluation harness scores column names. `samples/manifest.json`
  gains `expected.columns`, derived by `samples/generate_samples.py` from
  each fixture's header line (parsed by the stdlib `csv` module with the
  manifest dialect, names kept as written; `column_1..N` for a header-less
  file). `scripts/eval_samples.py` counts the exact list match in the
  score and reports per-name recall and column-count match as
  diagnostics next to each file, so a paraphrased name (`Monto` for
  `Importe`) no longer scores 100 %
  ([#123](https://github.com/deluispablo/data-agent-toolkit/issues/123)).
- Sixteen hard-case fixtures in `samples/`, each with a one-line note
  naming the grounding or sampling rule it guards: a header-less file
  after preamble lines, quoted line breaks in the head and in the tail
  window, a footer longer than the tail window, totals-shaped data rows
  (`2024,,4241.25`, `Total Energies SA`), decimal-comma, tab-with-commas
  and quoted-pipe dialects, a cp1252 name only in the tail, one-column,
  one-row and exactly-4096-byte files, duplicate and blank column names,
  a data value equal to a prompt marker, and a header of years. Four are
  flagged `known_limitation`. A header-less fixture's positional column
  names are now sized by its most common row width, so preamble lines do
  not set the count
  ([#125](https://github.com/deluispablo/data-agent-toolkit/issues/125)).
- The fixture catalog grows from 44 to 80 files. `samples/matrix.py`
  renders a table of `FixtureSpec`s (delimiter, quote character,
  encoding and BOM, line ending, preamble, header, footer kind, width,
  length, decimal comma, quoted header, ragged rows) into `gen_*.csv`
  fixtures with their full ground truth: every footer kind on a wide and a
  narrow file, every encoding with LF and CRLF, two files over 64 KiB,
  quirk combinations, a 40-column UTF-16 file and a known-limitation file
  whose lines are longer than the head window. Manifest entries gain a
  `generated` flag
  ([#124](https://github.com/deluispablo/data-agent-toolkit/issues/124)).
- The prompt is versioned: `result.usage.prompt_version` records the
  version of the prompt the models were sent (`2026.09-c` today), and the
  usage log line includes it, so measurements of different prompts are
  never mixed. Unit tests fail when the prompt template grows more than
  10 % past its measured size, and pin each prompt branch to a golden
  string ([#127](https://github.com/deluispablo/data-agent-toolkit/issues/127)).
- The evaluation harness writes machine-readable runs and compares them.
  `scripts/eval_samples.py --out` writes one JSON line per fixture and
  repeat (verdicts, `usage`, latency, errors per attempt, and the raw model
  text with `--keep-raw`) plus a summary line (harness and prompt
  versions, scores per category and field, token totals and means, latency
  p50/p95, error counts, retries, 429/503 seen). `--repeat N` adds
  majority-vote accuracy and agreement per field and lists fixtures whose
  answers drift; `--model` is repeatable. Quota guards for cloud runs:
  `--max-calls` (a hard stop counting fallbacks and retries), `--rpm`,
  `--max-fixtures`, `--fixture`, `--dry-run` (prompt sizes, no model call),
  and a warning with the planned call count when `--backend api` runs
  without `--max-calls`. The new `scripts/compare_runs.py` renders two or
  more runs as a Markdown table plus the fixtures whose verdict changed.
  `runs/` is git-ignored
  ([#122](https://github.com/deluispablo/data-agent-toolkit/issues/122)).
- Evaluation harness follow-ups (harness version 2). Run files start with
  a `run` line and stream one line per inspection as it finishes, so an
  interrupted run keeps its finished lines; `--summarize RUN` appends an
  `incomplete` summary to such a file, and `compare_runs.py` labels it
  "(incomplete, N/M fixtures)". `--subset quick` (21 fixtures, for
  iterating) and `--subset cloud` (the 15-fixture free-tier list) select
  documented lists. Manifest entries gain a generator-owned
  `expected_error`: `empty_file.csv`'s `EmptySampleError` now scores as a
  pass instead of a pipeline error. The summary's `retries` counts the
  retries of failed cloud inspections too, and `--dry-run` estimates
  tokens at the measured 1.81 characters per token
  ([#149](https://github.com/deluispablo/data-agent-toolkit/issues/149)).

### Changed

- The local backend sends the answer's JSON Schema to Ollama
  (`format=<schema>`, structured outputs) instead of plain JSON mode, and
  the cloud backend sends the same schema, stripped of titles,
  descriptions and defaults. The prompt no longer spells out the JSON
  shape, only what the fields mean: the instruction template shrinks from
  3,291 to 2,813 characters (`PROMPT_VERSION` `2026.09-c`). An Ollama
  server older than 0.5 that rejects a schema is asked again with
  `format="json"`, with a WARNING. No dependency change: `ollama` 0.6.2
  already accepts a schema
  ([#130](https://github.com/deluispablo/data-agent-toolkit/issues/130)).
- Grounding decides that a claimed header at row 0 is really data with a
  deterministic shape test on the sample instead of the model's
  `example_values`. When the model's names cannot be anchored in the head,
  row 0 is data if rows 0 and 1 have as many fields as inferred columns
  and their per-field shapes (integer, decimal with `.` or `,`, date,
  empty or text) are identical, or agree field by field (an empty cell
  matching any shape) with at least half the fields non-text in both
  rows. A row-0 field equal to one of the model's column names,
  case-insensitively, always keeps the header, and a header of years
  above decimal data stays a header. The result is then reported with
  `has_header=false` and positional names, as before
  ([#131](https://github.com/deluispablo/data-agent-toolkit/issues/131)).

### Fixed

- Wide files no longer fail on local models. On 0.3.0 every inspection of
  a 40-column file failed with both default Ollama models: the per-column
  answer (about 30 to 40 completion tokens per column) was cut off at the
  1024-token reply cap, so it was invalid JSON, and the fallback hit the
  same cap. With column names only, a 40-column answer fits well within
  the cap
  ([#147](https://github.com/deluispablo/data-agent-toolkit/issues/147)).
- Delimiter grounding replaces the model's delimiter when another usual
  delimiter clearly dominates it, not only when it splits fewer than two
  head lines: a `,` answered for a tab-separated file whose values hold
  commas (`"Fernández, Asociados"`, `1,50`) is now replaced by the tab
  when exactly one candidate splits at least twice as many head lines into
  the same number of fields. Ties, one-column files and exotic delimiters
  still keep the model's answer
  ([#151](https://github.com/deluispablo/data-agent-toolkit/issues/151)).
- Footer grounding no longer turns data rows into a footer. A data row
  (the modal field count of the end of the file, at least half of its
  fields filled, not a totals row) is never a footer line: the footer
  starts past any the model reported, and a footer made only of data rows
  is dropped with the existing WARNING. Marker and timestamp lines just
  above the reported footer line are now part of the footer, as blank and
  totals lines already were. The model's footer lines also anchor
  tolerantly: trailing empty fields (`TOTAL,,356681.99,` for
  `TOTAL,,356681.99,,`) are ignored, and a reported text of 8 characters or
  more matches the line it occurs in (a timestamp copied without its
  `Generado el` prefix). One-column files, and files whose delimiter does
  not split most lines, keep the previous rules
  ([#153](https://github.com/deluispablo/data-agent-toolkit/issues/153)).
- A header row with a blank name the model left out (`,id,id,value` answered
  as `id, id, value`) is now anchored: the names are taken from the file,
  blanks included (a blank name is reported as `""`)
  ([#153](https://github.com/deluispablo/data-agent-toolkit/issues/153)).
- A footer now always starts after the last data row: a ragged data row
  the model points at, with full rows after it, is no longer a footer, and
  a data row the model reports but that is not in the file (miscopied or
  made up) is read as "the footer starts after the data". A
  footer line the model copied with its own delimiter, replaced by
  grounding (`TOTAL,,12.50` in a tab-separated file), still anchors
  ([#129](https://github.com/deluispablo/data-agent-toolkit/issues/129),
  follow-up of [#153](https://github.com/deluispablo/data-agent-toolkit/issues/153)).

### Documentation

- New `docs/evaluation.md`: the before/after ritual for changes that can
  move accuracy, the flag reference, run file format, how `columns` is
  scored, quota notes, hand-written vs matrix fixtures, the known
  limitations, the prompt-change checklist and a fifteen-fixture cloud
  subset. Linked from the README and `ARCHITECTURE.md`
  ([#122](https://github.com/deluispablo/data-agent-toolkit/issues/122)).
- `docs/evaluation.md` records the 0.3.0 baseline on the 80-fixture
  catalog: prompt and completion token breakdowns, accuracy per category
  and field, latency and reloads, failure modes, a cloud run with its list
  price, and the GO/RESCOPE/DROP verdict for each planned optimization
  ([#128](https://github.com/deluispablo/data-agent-toolkit/issues/128)).

## [0.3.0] - 2026-09-24

### Changed (breaking)

- The result can describe a header-less file: new field
  `has_header: bool` (default `true`), and `header_row_index` is now
  `int | None`, `None` exactly when `has_header` is `false`. A model answer
  of `null` or `-1` without `has_header` is read as "no header row"; a
  contradiction fails validation. Grounding keeps the positional column
  names (`column_1`, ...) of a header-less file instead of anchoring a
  data row as the header, and corrects a header claimed at row 0 whose
  fields are the model's own example values to "no header". Consumers must handle `header_row_index=None`
  (for example `skiprows=result.header_row_index or 0`, `header=None`);
  see `docs/using-the-result.md`. A header-less file with preamble lines
  is not described (known limitation). The JSON contract gains
  `has_header`; the version moves to 0.3.0
  ([#94](https://github.com/deluispablo/data-agent-toolkit/issues/94)).
- The default cloud models are now `gemini-3.6-flash` (primary) and
  `gemini-flash-lite-latest` (fallback). The Gemini Developer API answers
  `404 NOT_FOUND` ("no longer available to new users") for the previous
  defaults, `gemini-2.5-flash` and `gemini-2.5-flash-lite`. The fallback is
  Google's moving Flash-Lite alias: it was the most reliable model on a
  free-tier key during verification, and it tracks the current Flash-Lite
  release. Pin a versioned name in `CLOUD_FALLBACK_MODEL` for reproducible
  results. Set `CLOUD_MODEL` / `CLOUD_FALLBACK_MODEL` (or the `Settings`
  fields) to keep the old names on projects that still have access
  ([#5](https://github.com/deluispablo/data-agent-toolkit/issues/5)).

### Added

- A truncated head sample now ends on its last line break before it
  reaches the model and grounding: the partial last row, and a character
  cut in half with it (a trailing `U+FFFD` in UTF-16), are dropped. A head
  that covers the whole file, or has no line break, is unchanged. The
  prompt is at most one partial line shorter
  ([#102](https://github.com/deluispablo/data-agent-toolkit/issues/102)).
- `Settings.ollama_host` (environment variable `OLLAMA_HOST`, the name the
  Ollama SDK already uses) sets the Ollama server's base URL, so hosts
  configure the local backend through `Settings` instead of the process
  environment. Unset keeps the SDK default
  ([#96](https://github.com/deluispablo/data-agent-toolkit/issues/96)).
- Public exports `DEFAULT_SAMPLE_BYTES`, `DEFAULT_TAIL_BYTES`,
  `MAX_SAMPLE_BYTES`, `ModelInvoker` and `AsyncModelInvoker`, so hosts can
  validate sample windows and type their `model_invoker` seam without
  copying the library's values
  ([#100](https://github.com/deluispablo/data-agent-toolkit/issues/100)).

### Changed

- Internal cleanups; CLI range errors for `--bytes` / `--tail-bytes` now
  come from argparse converters (`expected an integer <= 16384`), and the
  CLI prints the result with `model_dump_json`
  ([#103](https://github.com/deluispablo/data-agent-toolkit/issues/103)).
- `load_settings()` reads the environment (and an explicit `.env` file)
  with the standard library instead of `pydantic-settings`, which is no
  longer a dependency of the `[cloud]` extra. A base install now reads
  `OLLAMA_MODEL` and the other variables, and the CLI reads `./.env`
  without the extra. The `.env` syntax is `KEY=VALUE` with comments,
  `export` and quotes; variable interpolation and multi-line values are
  not supported. The signature of `load_settings` is unchanged
  ([#99](https://github.com/deluispablo/data-agent-toolkit/issues/99)).
- With a fallback model, the primary may now use about 70 % of the
  `timeout_seconds` budget instead of half; the fallback still gets
  everything left. A cold 7B load on CPU often needed more than half, so
  the weaker fallback answered on every cold start. Trade-off: a hung
  primary now leaves the fallback about 30 % of the budget instead of
  50 %; on CPU-only local deployments of the API example, set
  `CSV_INSPECTOR_API_DEFAULT_TIMEOUT_SECONDS` to 90 or more. A model the
  budget leaves out is logged at INFO
  ([#95](https://github.com/deluispablo/data-agent-toolkit/issues/95)).
- The `api` backend retries a `429 RESOURCE_EXHAUSTED` or
  `503 UNAVAILABLE` answer once on the same model before falling back:
  after the `Retry-After` header (at most 10 s) or about one second, and
  only within the model's time budget. The retry is logged at WARNING
  ([#98](https://github.com/deluispablo/data-agent-toolkit/issues/98)).

### Fixed

- Delimiter grounding scores the model's delimiter like the candidates
  instead of keeping any delimiter that occurs in the head: a `,` answered
  for a tab-separated file whose values hold a comma (`1,5`,
  `"Smith, John"`) is now replaced by the tab. The model's answer is still
  kept on ties and for one-column files; a replacement is logged at INFO
  ([#97](https://github.com/deluispablo/data-agent-toolkit/issues/97)).
- A `quotechar` answered as `""`, `"null"`, `"none"` or JSON `null` (the
  natural answer for an unquoted file) now maps to the default `'"'` instead
  of failing validation and triggering the fallback model. The prompt asks
  for `null` when fields are never quoted
  ([#93](https://github.com/deluispablo/data-agent-toolkit/issues/93)).
- An empty Gemini response now says why in its `ModelInvocationError`: the
  prompt block reason or the candidate's finish reason (for example
  `SAFETY`).
- The `api` backend disables the SDK's automatic function calling, which it
  never used, so `google-genai` no longer logs an INFO line and a WARNING on
  every call.

### Documentation

- Known limitations are documented: the prompt's plain-text sample
  markers (bounded by grounding), the 64 MiB forward-scan limit of
  non-seekable streams, and header-less files with preamble lines
  ([#105](https://github.com/deluispablo/data-agent-toolkit/issues/105)).
- The grounding rules moved from the README to
  `docs/using-the-result.md` ("How the result is grounded"); the README
  keeps a one-paragraph summary
  ([#104](https://github.com/deluispablo/data-agent-toolkit/issues/104)).
- The `api` backend is verified against the real Gemini Developer API
  (`google-genai` 2.25.0): the `CSVInspectionResult` JSON Schema is
  accepted, timeouts map to `ModelTimeoutError`, and an invalid key maps to
  a redacted `ModelInvocationError`
  ([#5](https://github.com/deluispablo/data-agent-toolkit/issues/5)). Vertex
  AI is still unverified
  ([#92](https://github.com/deluispablo/data-agent-toolkit/issues/92)).

## [0.2.0] - 2026-09-24

### Changed (breaking)

- `ColumnSchema.inferred_type` is now a closed vocabulary, the new public
  `ColumnType` literal: `string`, `integer`, `float`, `date`, `datetime` or
  `boolean` (it was any string the model wrote). Common aliases are mapped
  (`int` and `bigint` to `integer`, `number` and `decimal` to `float`,
  `text` and `varchar` to `string`, `bool` to `boolean`, `timestamp` to
  `datetime`, and so on), and any other word becomes `string`. The prompt now
  offers `datetime`, and the JSON schema sent to Gemini lists the vocabulary
  as an enum. It also asks for at most 3 example values per column, since
  `qwen2.5-coder:7b` otherwise listed values without end on some files
  ([#35](https://github.com/deluispablo/data-agent-toolkit/issues/35)).
- The default fallback models are now `qwen2.5-coder:3b` (local) and
  `gemini-2.5-flash-lite` (cloud). They used to equal the primary models,
  so the fallback was skipped and only one model was ever tried. The local
  fallback must be pulled (`ollama pull qwen2.5-coder:3b`); set
  `OLLAMA_FALLBACK_MODEL` / `CLOUD_FALLBACK_MODEL` (or the `Settings`
  fields) to the primary to keep the previous single-model behaviour
  ([#13](https://github.com/deluispablo/data-agent-toolkit/issues/13)).
- `n_bytes` and `tail_bytes` (and the CLI's `--bytes` / `--tail-bytes`) are
  now capped at 16384 bytes each; larger values raise `ValueError` (a usage
  error in the CLI) ([#9](https://github.com/deluispablo/data-agent-toolkit/issues/9)).

### Added

- `ensure_backend_ready(backend, settings=None)` is now public. It is the
  configuration check every inspection already runs (cloud credentials,
  `google-genai` installed; no network or model call), so hosts can fail
  fast at startup or in a readiness probe
  ([#82](https://github.com/deluispablo/data-agent-toolkit/issues/82)).
- CLI option `--fallback-model`, the model tried after `--model` fails
  (default: the backend's configured fallback). The "Inspecting ..." log
  line now names both models
  ([#39](https://github.com/deluispablo/data-agent-toolkit/issues/39)).
- The CLI now has a default `--timeout` of 300 seconds, so a stalled Ollama
  ends in an `InspectionTimeoutError` message instead of a hang;
  `--timeout 0` removes the limit. The library default stays `None`
  ([#39](https://github.com/deluispablo/data-agent-toolkit/issues/39)).
- New guide, `docs/using-the-result.md`, on turning a result into reader
  options for the stdlib `csv` module, pandas and PySpark, with encoding
  names for Spark and BigQuery. The stdlib recipe is run by the test suite
  ([#36](https://github.com/deluispablo/data-agent-toolkit/issues/36)).
- A model answer that wraps its JSON object in prose rather than a code
  fence (`Here is the result: {...}`) is now parsed from the first `{` to
  the last `}` instead of failing as invalid JSON and using up an attempt
  ([#23](https://github.com/deluispablo/data-agent-toolkit/issues/23)).
- Sampling a non-seekable stream reads at most 64 MiB past the head while
  looking for its end. A longer stream gets no tail sample and is treated
  like `tail_bytes=0` (the end is unsampled, so no footer is reported),
  instead of being read to the end in unbounded time
  ([#17](https://github.com/deluispablo/data-agent-toolkit/issues/17)).
- The delimiter is now grounded in the head sample. When the model reports
  one that never occurs there (small models answer `,` for tab-separated
  files), the usual delimiter (`,`, `;`, tab, `|`) that splits the most
  lines into the same number of fields is used instead. A delimiter that
  occurs is never changed
  ([#53](https://github.com/deluispablo/data-agent-toolkit/issues/53)).

### Fixed

- The configured Gemini API key is redacted (`***`) from the `WARNING` logged
  for a failed model attempt, so a custom `model_invoker` whose error message
  contains the key no longer leaks it into the host's logs
  ([#81](https://github.com/deluispablo/data-agent-toolkit/issues/81)).
- When the last model was cut at `timeout_seconds`, a timed wait that
  returned a hair before the clock reached the deadline (seen on Windows)
  surfaced as `InspectionFailedError` instead of `InspectionTimeoutError`
  ([#57](https://github.com/deluispablo/data-agent-toolkit/issues/57)).
- A model answer whose dialect characters conflict (the delimiter equal to
  the quote or escape character, or a line break as a dialect character)
  made grounding raise a raw `ValueError` from the `csv` module, so the
  fallback model never ran. Such an answer now fails validation with
  `SchemaValidationError` and the fallback model runs. An escape character
  equal to the quote character is read as doubled quotes
  (`escapechar=None`, `doublequote=True`)
  ([#48](https://github.com/deluispablo/data-agent-toolkit/issues/48)).
- The `ollama` floor was `>=0.6`, but `ollama` 0.6.0 and 0.6.1 cannot be
  used as a context manager, so with them every local model call failed.
  The floor is now `ollama>=0.6.2`, and CI tests the lowest allowed version
  of every direct dependency
  ([#49](https://github.com/deluispablo/data-agent-toolkit/issues/49)).
- A reported footer that does not occur at the sampled end of the file was
  kept as reported, so `footer_rows_to_skip` made readers silently drop real
  data rows. It is now discarded (`footer_lines == []`) with a warning
  ([#52](https://github.com/deluispablo/data-agent-toolkit/issues/52)).
- Ollama requests did not cap the reply length, so a model stuck repeating
  itself (for example an endless `example_values` list) generated until the
  timeout, or forever with no timeout. The reply is now capped at the 1024
  tokens the context window reserves for it; a truncated answer fails
  parsing and the fallback model runs.
- On a base install (no `[cloud]` extra), the CLI failed on every run when
  the working directory held a `.env` file. The implicit `./.env` is now
  ignored with a warning; an explicit `--env-file` still fails
  ([#33](https://github.com/deluispablo/data-agent-toolkit/issues/33)).
- The encoding was detected from the head sample only, so a file with a
  pure-ASCII head and cp1252/latin-1 bytes further down was reported as
  `utf-8` and its tail decoded with replacement characters. When the tail is
  not valid UTF-8, the encoding is now detected from the head and tail
  together ([#34](https://github.com/deluispablo/data-agent-toolkit/issues/34)).
- Grounded column names were stripped of surrounding spaces, so they did not
  match the names `csv` and pandas read. They now keep the header fields
  exactly as written; matching still ignores the padding
  ([#37](https://github.com/deluispablo/data-agent-toolkit/issues/37)).
- With `tail_bytes=0`, a path, buffer or seekable stream of exactly
  `n_bytes` was reported as truncated and lost its footer. Its known size now
  tells a complete head from a truncated one; non-seekable streams keep the
  old assumption
  ([#38](https://github.com/deluispablo/data-agent-toolkit/issues/38)).
- With `tail_bytes=0` on a source larger than the head, the head was
  presented to the model as the entire file, so its last (possibly truncated)
  rows could be reported and grounded as footer, and consumers skipped real
  data. The end of the source is now known to be unsampled: the prompt says
  so and `footer_lines` is left empty
  ([#10](https://github.com/deluispablo/data-agent-toolkit/issues/10)).
- Ollama requests never set `num_ctx`, so a prompt larger than the server's
  small default context window lost its start (the instructions and head
  sample) and the model answered garbage that still passed validation. The
  window is now sized from the prompt
  ([#9](https://github.com/deluispablo/data-agent-toolkit/issues/9)).
- `delimiter`, `quotechar` and `escapechar` were not validated, so answers
  such as `"\\t"`, `"tab"` or `""` passed and broke `csv`/pandas
  downstream. Common spellings of a tab now become a real tab, `""` /
  `"null"` mean no `escapechar`, and anything else longer or shorter than
  one character is a `SchemaValidationError`, so the fallback model runs
  ([#11](https://github.com/deluispablo/data-agent-toolkit/issues/11)).
- The primary model could spend the whole `timeout_seconds` budget, so a
  hung or slowly loading primary meant the fallback was never tried. Each
  model now gets an equal share of what is left, and unused time carries
  over ([#12](https://github.com/deluispablo/data-agent-toolkit/issues/12)).
- An exception other than the library's own from a custom `model_invoker`
  (`RuntimeError`, raw `httpx` errors, `KeyError`...) skipped the fallback
  and escaped unwrapped. It now counts as a failed attempt, and
  `InspectionFailedError` is raised if every model fails, in the sync and
  async APIs alike
  ([#14](https://github.com/deluispablo/data-agent-toolkit/issues/14)).
- A sync call abandoned by `timeout_seconds` left a non-daemon worker
  thread, so a custom invoker that never returned kept the interpreter
  from exiting. The worker is now a daemon thread
  ([#15](https://github.com/deluispablo/data-agent-toolkit/issues/15)).
- Grounding split lines with `str.splitlines()`, which also breaks on form
  feeds, `\x1c`-`\x1e`, `\x85`, U+2028 and U+2029. Such characters in the
  data shifted `header_row_index` and could pull data into
  `footer_lines`. Lines are now split only on `\r\n`, `\r` and `\n`, like
  `csv` and pandas
  ([#16](https://github.com/deluispablo/data-agent-toolkit/issues/16)).
- Sampling a seekable stream relied on `seek()` returning the new
  position, so duck-typed streams whose `seek()` returns `None` failed
  with a `TypeError`
  ([#18](https://github.com/deluispablo/data-agent-toolkit/issues/18)).
- A non-blocking stream whose `read()` returned `None` (no data yet) was
  taken as ended, so the sample was silently truncated or
  `EmptySampleError` was raised. It is now a `FileSampleReadError`
  ([#19](https://github.com/deluispablo/data-agent-toolkit/issues/19)).
- A data row whose first field starts with a totals word (e.g. `Total
  Energies,2024-01-01,10.00`) just above the footer was pulled into
  `footer_lines`. A totals row must now have a bare label as its first
  field (`TOTAL`, `Subtotal:`) or mostly empty other fields
  ([#20](https://github.com/deluispablo/data-agent-toolkit/issues/20)).
- When the model paraphrased column names, the header search also matched
  on empty names, so with an unnamed column (e.g. a pandas index) any data
  row with an empty cell could be taken as the header
  ([#21](https://github.com/deluispablo/data-agent-toolkit/issues/21)).
- The model's `encoding` was used unchecked: it could be a description
  rather than a codec (`"UTF-8 with BOM"`), or drop a detected byte order
  mark (`utf-8` for a `utf-8-sig` file), leaving a U+FEFF on the first
  column name downstream. An encoding detected from a BOM now always
  wins, and an unknown codec name falls back to the detected encoding
  ([#22](https://github.com/deluispablo/data-agent-toolkit/issues/22)).

## [0.1.0] - 2026-09-24

First release as an installable, embeddable library
([#7](https://github.com/deluispablo/data-agent-toolkit/issues/7)).
Earlier work in the monorepo is folded into this release.

### Added

- **Installable package** `csv-inspector`: src layout, `pyproject.toml`
  (hatchling), `py.typed`, a `[cloud]` extra (`google-genai`,
  `pydantic-settings`), installable from a path or from git
  (`#subdirectory=agents/csv_inspector`).
- **Explicit public API** in `csv_inspector.__all__`: `inspect_csv`,
  `ainspect_csv`, `CSVInspectionResult`, `ColumnSchema`, `LLMBackend`,
  `Settings`, `load_settings`, `CSVSource`, `__version__` and the exception
  hierarchy. Everything else is internal.
- **Bytes and stream input**: `inspect_csv` accepts a path, `bytes`,
  `bytearray`, `memoryview`, or any binary file-like object. Seekable
  streams are sampled from their current position, which is then
  restored; non-seekable streams are read once, with memory bounded by
  `n_bytes + tail_bytes`; text streams are rejected with `TypeError`.
- **Async API**: `ainspect_csv`, using `ollama.AsyncClient` and
  `google-genai`'s `client.aio`. Sampling runs in a worker thread.
- **Timeouts**: `timeout_seconds`, one overall budget shared by the primary
  and fallback models, enforced by the library (custom invokers included)
  and passed to the HTTP clients (milliseconds for Gemini). New
  `InspectionTimeoutError` (a subclass of `InspectionFailedError`) and
  `ModelTimeoutError` (a subclass of `ModelInvocationError`).
- **Injectable settings**: `Settings` is a plain, frozen Pydantic model that
  never reads the environment and rejects unknown fields; pass it with
  `settings=`. `load_settings(env_file=...)` reads the environment
  explicitly, and a `.env` file only when asked.
- **CLI** in the package: the `csv-inspector` console script and
  `python -m csv_inspector`, with `--timeout`, `--env-file` and
  `--no-env-file`.
- Selectable backend: local Ollama (default) or Google Gemini (Gemini
  Developer API or Vertex AI) via `backend=LLMBackend.API`. The Gemini
  backend is unit-tested with a mocked client, and its end-to-end
  verification is pending
  ([#5](https://github.com/deluispablo/data-agent-toolkit/issues/5)).
- Footer detection with grounding in the sampled text (header row, literal
  column names, verbatim footer lines including blank and totals rows);
  `footer_rows_to_skip` derived from `footer_lines`.
- Bounded head/tail sampling with non-overlapping windows, UTF-16/UTF-32
  alignment and validated byte budgets.

### Changed (breaking)

Relative to the unpackaged monorepo code:

- **Imports**: `from inspector import inspect_csv` (and `models`,
  `exceptions`, ...) becomes `from csv_inspector import ...`.
- **Installation**: `requirements.txt` / `requirements-cloud.txt` are
  replaced by `pip install ./agents/csv_inspector` or
  `pip install "./agents/csv_inspector[cloud]"`.
- **`.env` is no longer read implicitly** by the library; use
  `load_settings(env_file=...)`, or the CLI, which still reads `./.env` by
  default.
- `inspect_csv`'s first parameter is now the **positional-only** `source`
  (previously `path`).
- Logger names are now under `csv_inspector.*`.
- A missing `ollama` package raises `BackendConfigurationError` (still a
  `ModelInvocationError`).

### Removed

- The `metadata_lines` field of `CSVInspectionResult`; the preamble is
  described by `header_row_index`.

[Unreleased]: https://github.com/deluispablo/data-agent-toolkit/compare/csv-inspector-v0.3.0...HEAD
[0.3.0]: https://github.com/deluispablo/data-agent-toolkit/compare/csv-inspector-v0.2.0...csv-inspector-v0.3.0
[0.2.0]: https://github.com/deluispablo/data-agent-toolkit/compare/csv-inspector-v0.1.0...csv-inspector-v0.2.0
[0.1.0]: https://github.com/deluispablo/data-agent-toolkit/releases/tag/csv-inspector-v0.1.0
