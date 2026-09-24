# Changelog

All notable changes to `csv-inspector` are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project adheres to [Semantic Versioning](https://semver.org/). While the
version is `0.x`, minor releases may include breaking changes; each one is
listed under **Changed (breaking)**.

## [Unreleased]

### Added

- A model answer that wraps its JSON object in prose rather than a code
  fence (`Here is the result: {...}`) is now parsed from the first `{` to
  the last `}` instead of failing as invalid JSON and using up an attempt
  ([#23](https://github.com/deluispablo/data-agent-toolkit/issues/23)).
- Sampling a non-seekable stream reads at most 64 MiB past the head while
  looking for its end. A longer stream gets no tail sample and is treated
  like `tail_bytes=0` (the end is unsampled, so no footer is reported),
  instead of being read to the end in unbounded time
  ([#17](https://github.com/deluispablo/data-agent-toolkit/issues/17)).

### Fixed

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

### Changed (breaking)

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

[Unreleased]: https://github.com/deluispablo/data-agent-toolkit/compare/csv-inspector-v0.1.0...HEAD
[0.1.0]: https://github.com/deluispablo/data-agent-toolkit/releases/tag/csv-inspector-v0.1.0
