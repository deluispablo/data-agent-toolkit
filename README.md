# data-agent-toolkit

[![CI](https://github.com/deluispablo/data-agent-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/deluispablo/data-agent-toolkit/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A monorepo of agentic AI utilities for data engineering, built in Python and
designed to run **entirely for free, locally**, via
[Ollama](https://ollama.com), with optional cloud/API backends (e.g. Gemini)
as an opt-in. Each agent is an **installable, embeddable library** that you
add as a dependency to your own application or API.

## Agents

| Agent | Package | What it does | Docs |
|---|---|---|---|
| `csv_inspector` | `csv-inspector` 0.1.0 | Infers the encoding, dialect, header row, footer lines and column schema of large, messy CSV/TSV sources from bounded head/tail samples. Accepts paths, bytes or streams; sync and async API. | [README](agents/csv_inspector/README.md) · [Embedding guide](agents/csv_inspector/docs/embedding.md) · [Changelog](agents/csv_inspector/CHANGELOG.md) |

Using an agent in another project:

```bash
pip install "csv-inspector @ git+https://github.com/deluispablo/data-agent-toolkit@csv-inspector-v0.1.0#subdirectory=agents/csv_inspector"
```

```python
from csv_inspector import inspect_csv

result = inspect_csv(uploaded_bytes)
```

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
5. **Embeddable libraries, not services.** Each agent is a pip-installable
   package with an explicit public API, injectable configuration, sync and
   async entry points, bounded latency, and library-grade logging. The host
   application owns HTTP, deployment and secrets.

## Repository layout

```
data-agent-toolkit/
├── .github/workflows/ci.yml        # Lint, type-check, test matrix, packaging checks
├── agents/
│   └── csv_inspector/              # The csv-inspector package (see its README)
│       ├── pyproject.toml          # Package metadata, dependencies, [cloud] extra, CLI entry point
│       ├── src/csv_inspector/      # The library (public API in __init__.py; _modules are internal)
│       ├── docs/embedding.md       # How to embed it in a host application
│       ├── CHANGELOG.md
│       ├── README.md               # Package README (also the PyPI long description)
│       ├── main_demo.py            # Repo demo: the CLI on the bundled sample.csv
│       ├── sample.csv              # Synthetic "dirty" CSV fixture
│       ├── samples/                # 28 generated edge-case fixtures + ground-truth manifest
│       └── scripts/
│           ├── eval_samples.py         # Manual LLM evaluation harness (not run in CI)
│           └── smoke_test_installed.py # Post-install smoke test used by CI
├── tests/                          # Test suite for all agents (runs against the installed package)
├── .env.example                    # Settings template for the CLIs
├── .gitattributes                  # LF for sources; CSV/TSV fixtures kept byte-exact
├── pyproject.toml                  # Repository-wide ruff, mypy and pytest configuration
├── requirements-dev.txt            # Editable install of every agent + dev tooling
└── LICENSE
```

## Development

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements-dev.txt   # editable csv-inspector[cloud] + pytest, ruff, mypy, build, twine

ollama serve                          # only needed for live runs, never for tests
ollama pull qwen2.5-coder:7b
ollama pull qwen2.5-coder:3b          # default fallback model

python agents/csv_inspector/main_demo.py              # or: csv-inspector path/to/file.csv
python agents/csv_inspector/scripts/eval_samples.py   # score a live model on the fixture catalog
```

The agents are installed in **editable** mode, so the tests and scripts
import them exactly as an external host would (`import csv_inspector`),
with no `sys.path` manipulation for library code. Only repository tooling
that is deliberately not packaged (the fixture generator, the evaluation
harness) is put on the path by `tests/conftest.py`.

The CLIs read settings from environment variables and, as applications,
from `./.env` when present (copy [`.env.example`](.env.example)). The
libraries themselves never read a `.env` file implicitly.

## Testing and quality checks

```bash
ruff check .          # lint
ruff format --check . # formatting
mypy                  # strict static type-check (with the pydantic plugin)
pytest                # tests: no Ollama, no credentials, no network
```

[CI](.github/workflows/ci.yml) runs on every push and pull request:

- **Lint and type-check.**
- **Tests** on Python 3.10–3.14 (Linux), plus Windows, which also guards
  the byte-exact fixtures against CRLF conversion.
- **Packaging.** It builds the sdist and wheel, runs `twine check`,
  installs the wheel into a **fresh virtual environment** (without the
  `[cloud]` extra), and smoke-tests the public API **from outside the
  repository**. It also installs the sdist with `[cloud]`.

The test suite is hermetic. An autouse fixture clears every settings
variable and runs each test from an empty directory, and every model
backend is faked (Ollama is replaced by a stand-in module; the Gemini
client by a recorder that keeps the SDK's real request types). The one
thing intentionally not covered by `pytest` is the LLM's accuracy itself.
That is non-deterministic, so it is measured manually with
`scripts/eval_samples.py` against a live model.

## Code standards

- 100% English source code, comments, and docstrings.
- [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
  on every public module, class, and function: arguments, return values, and
  raised exceptions are documented explicitly.
- Full static type hints (`typing` / PEP 604 union syntax) on all functions
  and classes, checked with `mypy --strict`. Packages ship `py.typed`.
- PEP 8 / PEP 257 compliance and formatting enforced by `ruff` (lint +
  format, 100-column lines), configured in `pyproject.toml`.
- **Library code never configures logging or prints**: it logs under its
  package logger with only a `NullHandler`. `logging.basicConfig()` and
  `print()` live in the CLI layer only (enforced by a test).
- Explicit, typed, domain-specific exception hierarchies instead of bare
  `Exception` handling.
- Structured, Pydantic-validated output for every agent, designed for
  direct consumption by downstream pipelines.

## Adding a new agent

1. Create `agents/<agent_name>/` as a package: `pyproject.toml`,
   `src/<package>/` with a `py.typed` marker and an explicit `__all__` in
   `__init__.py` (internal modules prefixed with `_`), `README.md` and
   `CHANGELOG.md`.
2. Default to local, free execution via Ollama; make cloud/API backends an
   opt-in extra with lazily imported SDKs, and accept injected settings.
3. Return Pydantic-validated structured output, never free text.
4. Add its editable install to `requirements-dev.txt`, its `src` to
   `mypy_path` and `files`, and its import name to `known-first-party` in
   the root `pyproject.toml`; add a packaging job for it in CI.
5. Add `tests/test_<agent_name>*.py`, faking every external backend so the
   suite runs without credentials or network access.
6. Add it to the agents table above.

## License

Released under the [MIT License](LICENSE).
