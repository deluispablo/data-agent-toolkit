# data-agent-toolkit

[![CI](https://github.com/deluispablo/data-agent-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/deluispablo/data-agent-toolkit/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Production-grade monorepo of agentic AI utilities for data engineering, built
in Python and designed to run **entirely for free, locally**, via
[Ollama](https://ollama.com), with optional cloud/API backends (e.g. Gemini)
as an opt-in.

## Design principles

1. **Local-first, dual-LLM.** Every agent defaults to a free, local Ollama
   model. Cloud/API backends are supported but never required to run the
   demo or the tests.
2. **Stateless, byte-budgeted inspection.** Agents that work on files never
   load them fully into memory; they operate on small, explicit byte
   samples, keeping cost and memory overhead close to zero even against
   multi-gigabyte sources.
3. **Structured, validated output.** Every agent returns a
   [Pydantic v2](https://docs.pydantic.dev/) model, not free text, so results
   plug directly into downstream pipelines (PySpark, BigQuery, Dataform,
   Airflow, Cloud Run, etc.).
4. **Explicit failure modes.** Each agent defines its own exception
   hierarchy rooted at a single domain base exception, instead of leaking
   bare `Exception`s to callers.
5. **Self-contained and testable.** Each agent folder ships its own
   dependencies, a synthetic data fixture, a local demo script, and can be
   exercised end-to-end without cloud credentials.

## Repository layout

```
data-agent-toolkit/
├── .github/workflows/ci.yml    # Lint, type-check and test matrix on every push/PR
├── agents/
│   └── csv_inspector/
│       ├── inspector.py        # Core inspection pipeline (head + tail sampling)
│       ├── models.py           # Pydantic v2 schema (structured output contract)
│       ├── exceptions.py       # Domain exception hierarchy
│       ├── main_demo.py        # Local, credential-free demo/CLI
│       ├── eval_samples.py     # Manual LLM evaluation harness (not run in CI)
│       ├── cli_support.py      # Shared CLI plumbing (arg types, logging, UTF-8 stdout)
│       ├── sample.csv          # Original synthetic "dirty" CSV fixture
│       ├── samples/            # Catalog of 28 dialect/structural edge-case fixtures
│       │   ├── generate_samples.py  # Reproducibly (re)generates every fixture below
│       │   ├── manifest.json        # Ground truth per fixture, used by tests + eval_samples.py
│       │   └── *.csv / *.tsv        # The generated fixtures themselves
│       └── requirements.txt    # Runtime dependencies for this agent
├── common/                     # Shared utilities across agents (as they emerge)
├── tests/
│   ├── conftest.py             # Adds each agent's directory to sys.path
│   ├── test_csv_inspector.py   # Unit tests: byte sampling, prompt building, orchestration
│   ├── test_samples_catalog.py # Deterministic tests parametrized over samples/ (no LLM)
│   ├── test_eval_samples.py    # Scoring logic of the evaluation harness (no LLM)
│   └── test_cli_support.py     # CLI argument helpers
├── .gitattributes              # LF for sources; CSV/TSV fixtures kept byte-exact
├── pyproject.toml              # ruff, mypy and pytest configuration
├── requirements-dev.txt        # Dev dependencies (pytest, ruff, mypy, + runtime deps)
├── LICENSE
└── README.md
```

## Agents

### `csv_inspector`

A stateless agent that inspects only a bounded **head** window (first N
bytes) and, for files larger than that window, a bounded **tail** window
(last N bytes) of a large, potentially messy CSV/TSV file — never loading
the full file into memory, in either direction — and uses a local LLM (via
Ollama) to infer:

- Character encoding
- Field delimiter, quote character, escape rules
- Non-standard header lines (export banners, comments) *and* footer lines
  (totals rows, "end of report" markers), sourced respectively from the
  head and tail samples
- A preliminary column-level schema (name, inferred type, nullability,
  example values)

The result is returned as a Pydantic-validated `CSVInspectionResult`, ready
to drive downstream ingestion logic.

#### Pipeline

```mermaid
flowchart TD
    A[Source CSV / TSV file] --> B[read_sample_bytes:<br/>bounded read of first n_bytes]
    B -->|empty file| X[EmptySampleError<br/>no LLM call]
    B --> C[chardet: heuristic encoding detection]
    C --> D[Decode head sample to text]
    B --> COND{Bytes left past<br/>the head window?}
    COND -->|no: skip tail, save tokens| E1[Build prompt: head only]
    COND -->|yes| T[read_tail_bytes:<br/>seek from end, bounded read of the<br/>uncovered bytes, max tail_bytes]
    T --> T2[Decode tail sample to text<br/>BOM-less codec, code-unit aligned;<br/>may start mid-line]
    D --> E1
    D --> E2[Build prompt: head + tail<br/>with mid-line caveat for the model]
    T2 --> E2
    E1 --> F{Invoke primary Ollama model}
    E2 --> F
    F -->|success: valid JSON + schema| L[CSVInspectionResult<br/>incl. footer_lines]
    F -->|failure: unreachable, bad JSON,<br/>or schema mismatch| G{Invoke fallback model}
    G -->|success: valid JSON + schema| L
    G -->|failure| H[InspectionFailedError<br/>aggregated per-model attempts]

    style L fill:#2f9e44,color:#fff
    style H fill:#c92a2a,color:#fff
    style X fill:#c92a2a,color:#fff
```

The tail window never overlaps the head: it covers at most the bytes the
head did not already read, and is skipped entirely when the head exhausts
the file — sending the same bytes twice would only waste tokens, in line
with this project's cost-optimization-first principle. For UTF-16/UTF-32
files the tail window is aligned to the code unit and decoded with the
explicit byte order taken from the head's BOM, so it never decodes as
garbage. When a tail sample is present, the prompt explicitly warns the
model that its first visible line is a blind byte-suffix and may be a
truncated fragment, not a real row.

Byte budgets are validated up front (`n_bytes >= 1`, `tail_bytes >= 0`):
a negative size passed to `file.read()` means "read everything", which is
exactly what this agent promises never to do.

#### Module layout

| File | Responsibility |
|---|---|
| `models.py` | `ColumnSchema`, `CSVInspectionResult` — the strict Pydantic v2 output contract, including `metadata_lines`/`footer_lines`. |
| `exceptions.py` | `CSVInspectorError` and its subclasses — one per domain failure mode. |
| `inspector.py` | Head/tail byte sampling (`read_sample_bytes`, `read_tail_bytes`), encoding detection, prompt construction, model invocation, JSON parsing/validation, and the primary/fallback orchestration in `inspect_csv`. |
| `main_demo.py` | CLI entry point for local, credential-free verification against `sample.csv`. |
| `eval_samples.py` | Manual evaluation harness: runs the real pipeline (Ollama required) against every fixture in `samples/` and scores it against `samples/manifest.json`. Not part of `pytest`/CI — see [Sample catalog](#sample-catalog) below. |
| `cli_support.py` | Shared CLI plumbing for both scripts: validated byte-budget argument types, `--log-level`, logging setup, and UTF-8 stdout (so accented output renders on Windows consoles). |
| `sample.csv` | Original synthetic fixture with deliberately messy characteristics: semicolon delimiter, metadata banner, accented values, embedded delimiters and escaped quotes. |
| `samples/` | Catalog of 28 further fixtures covering delimiters, encodings, header/footer variants, quoting/escaping, structural anomalies, and data-format gotchas — see below. |

The LLM backend is injected through `inspect_csv`'s `model_invoker`
parameter (default: `invoke_ollama_model`, which lazily imports the
`ollama` package). This keeps the module importable — and the pipeline
fully unit-testable — without the `ollama` package installed, and makes it
trivial to point the same pipeline at a different backend (e.g. a Gemini
API client) for the project's dual-LLM story.

#### Sample catalog

`agents/csv_inspector/samples/` holds 28 fixtures generated by
`generate_samples.py` (run it to regenerate the catalog reproducibly), each
documented with its ground truth in `manifest.json`:

| Category | Covers |
|---|---|
| `delimiter` | Comma, semicolon, tab, pipe, and a delimiter character appearing legitimately inside a quoted field. |
| `encoding` | UTF-8 with BOM, Latin-1/cp1252 (not valid UTF-8), UTF-16LE with BOM (classic old-Excel export). |
| `header_footer` | Metadata banners before the header, no header at all, a duplicated header mid-file, footer totals rows, an "end of report" marker, and both header and footer combined. |
| `quoting` | Doubled (`""`) and backslash-escaped quotes, an embedded real newline inside a quoted field (flagged `known_limitation`), inconsistent quoting, and a trailing empty field. |
| `structural` | Ragged rows, mixed CRLF/LF line endings, no trailing newline at EOF, blank lines between rows, an empty (0-byte) file, and a header-only file with zero data rows. |
| `data_format` | European decimal-comma numeric formatting, mixed null representations (`NULL`, `N/A`, `-`, `NaN`, empty), and whitespace-padded fields. |

Fixtures marked `known_limitation: true` in the manifest (currently just
the embedded-newline case) document a real limit of byte-window sampling —
a quoted multi-line record can be cut mid-record by a head/tail window —
rather than a bug this iteration is expected to fix silently.

Fixtures are byte-exact test data (line endings, BOMs and encodings are
the point), so `.gitattributes` marks `*.csv`/`*.tsv` as non-text and git
never rewrites them. After editing `generate_samples.py`, regenerate the
catalog with `python agents/csv_inspector/samples/generate_samples.py`; the
test suite fails if the committed fixtures or manifest drift from it.

Two different test layers consume this catalog:

- `tests/test_samples_catalog.py` (pytest, CI-safe, no LLM): validates the
  byte-sampling/encoding-detection layer, manifest/filesystem consistency,
  and that every fixture matches the generator byte for byte.
- `eval_samples.py` (manual, requires Ollama): runs the real LLM pipeline
  against every fixture and reports a per-file and aggregate accuracy
  score — see [Usage](#usage) below.

#### Usage

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 2. Install runtime dependencies
pip install -r agents/csv_inspector/requirements.txt

# 3. Make sure Ollama is running locally with the target model pulled
ollama serve
ollama pull qwen2.5-coder:7b

# 4. Run the demo against the bundled sample.csv
python agents/csv_inspector/main_demo.py

# Or against your own file, with a different model, custom head/tail
# sample sizes (--tail-bytes 0 disables tail sampling), and verbose logging
python agents/csv_inspector/main_demo.py --file path/to/file.csv --model qwen2.5-coder:7b --bytes 8192 --tail-bytes 8192 --log-level DEBUG

# 5. (Optional) Score the model against the full 28-fixture sample catalog
python agents/csv_inspector/eval_samples.py --model qwen2.5-coder:7b
```

#### Error handling

All domain failures raise a subclass of `CSVInspectorError`
(`agents/csv_inspector/exceptions.py`):

| Exception | Raised when |
|---|---|
| `FileSampleReadError` | The source file cannot be read from disk. |
| `EmptySampleError` | The source file is empty; raised before any model is invoked. |
| `ModelInvocationError` | The backend is unreachable, the `ollama` package is missing, the model is not pulled locally, or it returns an empty message. |
| `ResponseParsingError` | The model's response is not valid JSON. |
| `SchemaValidationError` | The parsed JSON does not satisfy the `CSVInspectionResult` schema. |
| `InspectionFailedError` | Every configured model (primary + fallback) failed; carries an `attempts` mapping of model name → exception for diagnostics. |

Invalid arguments (e.g. `n_bytes=0` or a negative `tail_bytes`) are
programming errors rather than domain failures, and raise a plain
`ValueError`.

## Testing

Tests live at the repository root under `tests/`, mirrored per agent
(`tests/test_<agent_name>.py`). The `csv_inspector` suite is CI-safe and
free of any Ollama dependency:

- `test_csv_inspector.py` uses dependency injection (`model_invoker`) to
  exercise the full pipeline — including the primary/fallback path and
  every domain error branch — **without** requiring a running Ollama
  instance or network access. It also proves, via a `Path.open()` spy, that
  `read_sample_bytes`/`read_tail_bytes` never fall back to a whole-file
  read regardless of source file size, and uses a stand-in `ollama` module
  to cover `invoke_ollama_model` itself.
- `test_samples_catalog.py` runs the byte-sampling/encoding-detection layer
  against every fixture in `agents/csv_inspector/samples/` and checks the
  fixtures and manifest stay in sync with `generate_samples.py`.
- `test_eval_samples.py` and `test_cli_support.py` cover the harness's
  scoring logic and the shared CLI helpers.

```bash
pip install -r requirements-dev.txt

ruff check .          # lint
ruff format --check . # formatting
mypy                  # strict static type-check
pytest                # tests
```

The same four checks run in [CI](.github/workflows/ci.yml) on every push
and pull request, against Python 3.10–3.14 on Linux plus Windows (which
also guards the byte-exact fixtures against CRLF conversion).

The one piece intentionally **not** covered by `pytest` is the LLM's
inferred accuracy itself — that is non-deterministic and lives in
`eval_samples.py` (see [Sample catalog](#sample-catalog)), run manually
against a live Ollama instance.

## Code standards

- 100% English source code, comments, and docstrings.
- [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
  on every public module, class, and function: arguments, return values, and
  raised exceptions are documented explicitly.
- Full static type hints (`typing` / PEP 604 union syntax) on all functions
  and classes, checked with `mypy --strict`.
- PEP 8 / PEP 257 compliance and formatting enforced by `ruff` (lint +
  format, 100-column lines), configured in `pyproject.toml`.
- `logging` module only in library and CLI code — no `print()` outside of
  the final output of the CLI scripts (`main_demo.py`, `eval_samples.py`).
- Explicit, typed, domain-specific exception hierarchies instead of bare
  `Exception` handling.
- Structured, Pydantic-validated output for every agent, designed for
  direct consumption by downstream pipelines.

## Adding a new agent

1. Create `agents/<agent_name>/` with its own `requirements.txt`,
   `models.py`, `exceptions.py`, core module(s), and `main_demo.py`.
2. Default to local, free execution via Ollama; make cloud/API backends
   opt-in through a pluggable interface (see `model_invoker` in
   `csv_inspector` for the pattern).
3. Return Pydantic-validated structured output — never free text.
4. Add `tests/test_<agent_name>.py`, injecting fakes for any external
   backend so the suite runs without credentials or network access.
5. Register the agent's module names under `known-first-party` and its
   directory in `mypy_path` (`pyproject.toml`), and in `tests/conftest.py`.
6. Update this README's agent table and, if the pipeline is non-trivial,
   add a Mermaid diagram.

## License

Released under the [MIT License](LICENSE).
