# data-agent-toolkit

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
├── agents/
│   └── csv_inspector/
│       ├── inspector.py        # Core inspection pipeline
│       ├── models.py           # Pydantic v2 schema (structured output contract)
│       ├── exceptions.py       # Domain exception hierarchy
│       ├── main_demo.py        # Local, credential-free demo/CLI
│       ├── sample.csv          # Synthetic "dirty" CSV fixture
│       └── requirements.txt    # Runtime dependencies for this agent
├── common/                     # Shared utilities across agents (as they emerge)
├── tests/
│   ├── conftest.py             # Adds each agent's directory to sys.path
│   └── test_csv_inspector.py   # Unit tests for csv_inspector
├── requirements-dev.txt        # Test/dev dependencies (pytest, + runtime deps)
└── README.md
```

## Agents

### `csv_inspector`

A stateless agent that inspects only the **first N bytes** of a large,
potentially messy CSV/TSV file — never loading the full file into memory —
and uses a local LLM (via Ollama) to infer:

- Character encoding
- Field delimiter, quote character, escape rules
- Non-standard header/footer lines (export banners, comments, blank rows)
- A preliminary column-level schema (name, inferred type, nullability,
  example values)

The result is returned as a Pydantic-validated `CSVInspectionResult`, ready
to drive downstream ingestion logic.

#### Pipeline

```mermaid
flowchart TD
    A[Source CSV / TSV file] -->|read first N bytes only| B[Raw byte sample]
    B --> C[chardet: heuristic encoding detection]
    C --> D[Decode sample to text]
    D --> E[Build structured inspection prompt]
    E --> F{Invoke primary Ollama model}
    F -->|success: valid JSON + schema| L[CSVInspectionResult]
    F -->|failure: unreachable, bad JSON,<br/>or schema mismatch| G{Invoke fallback model}
    G -->|success: valid JSON + schema| L
    G -->|failure| H[InspectionFailedError<br/>aggregated per-model attempts]

    style L fill:#2f9e44,color:#fff
    style H fill:#c92a2a,color:#fff
```

#### Module layout

| File | Responsibility |
|---|---|
| `models.py` | `ColumnSchema`, `CSVInspectionResult` — the strict Pydantic v2 output contract. |
| `exceptions.py` | `CSVInspectorError` and its subclasses — one per domain failure mode. |
| `inspector.py` | Byte sampling, encoding detection, prompt construction, model invocation, JSON parsing/validation, and the primary/fallback orchestration in `inspect_csv`. |
| `main_demo.py` | CLI entry point for local, credential-free verification against `sample.csv`. |
| `sample.csv` | Synthetic fixture with deliberately messy characteristics: semicolon delimiter, two non-standard metadata/comment lines before the real header, accented column and field values, embedded delimiters and escaped quotes inside quoted fields. |

The LLM backend is injected through `inspect_csv`'s `model_invoker`
parameter (default: `invoke_ollama_model`, which lazily imports the
`ollama` package). This keeps the module importable — and the pipeline
fully unit-testable — without the `ollama` package installed, and makes it
trivial to point the same pipeline at a different backend (e.g. a Gemini
API client) for the project's dual-LLM story.

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

# Or against your own file, with a different model and verbose logging
python agents/csv_inspector/main_demo.py --file path/to/file.csv --model qwen2.5-coder:7b --bytes 8192 --log-level DEBUG
```

#### Error handling

All domain failures raise a subclass of `CSVInspectorError`
(`agents/csv_inspector/exceptions.py`):

| Exception | Raised when |
|---|---|
| `FileSampleReadError` | The source file cannot be read from disk. |
| `ModelInvocationError` | The backend is unreachable, the `ollama` package is missing, or the model is not pulled locally. |
| `ResponseParsingError` | The model's response is not valid JSON. |
| `SchemaValidationError` | The parsed JSON does not satisfy the `CSVInspectionResult` schema. |
| `InspectionFailedError` | Every configured model (primary + fallback) failed; carries an `attempts` mapping of model name → exception for diagnostics. |

## Testing

Tests live at the repository root under `tests/`, mirrored per agent
(`tests/test_<agent_name>.py`). They use dependency injection
(`model_invoker`) to exercise the full pipeline — including the
primary/fallback path and every domain error branch — **without** requiring
a running Ollama instance or network access.

```bash
pip install -r requirements-dev.txt
pytest
```

## Code standards

- 100% English source code, comments, and docstrings.
- [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
  on every public module, class, and function: arguments, return values, and
  raised exceptions are documented explicitly.
- Full static type hints (`typing` / PEP 604 union syntax) on all functions
  and classes — PEP 8, PEP 257, and PEP 484 compliant, formatted to `ruff` /
  `black` conventions.
- `logging` module only in library and CLI code — no `print()` outside of
  the final structured result printed by `main_demo.py`.
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
5. Update this README's agent table and, if the pipeline is non-trivial,
   add a Mermaid diagram.
