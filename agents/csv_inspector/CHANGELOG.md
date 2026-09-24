# Changelog

All notable changes to `csv-inspector` are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project adheres to [Semantic Versioning](https://semver.org/). While the
version is `0.x`, minor releases may include breaking changes; each one is
listed under **Changed (breaking)**.

## [Unreleased]

### Fixed

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

### Changed (breaking)

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
