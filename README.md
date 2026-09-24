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
| `csv_inspector` | `csv-inspector` 0.1.0 | Infers the encoding, dialect, header row, footer lines and column schema of large, messy CSV/TSV sources from bounded head/tail samples. Accepts paths, bytes or streams; sync and async API. | [README](agents/csv_inspector/README.md) · [Embedding guide](agents/csv_inspector/docs/embedding.md) · [Using the result](agents/csv_inspector/docs/using-the-result.md) · [Changelog](agents/csv_inspector/CHANGELOG.md) |

Using an agent in another project:

```bash
pip install "csv-inspector @ git+https://github.com/deluispablo/data-agent-toolkit@csv-inspector-v0.1.0#subdirectory=agents/csv_inspector"
```

```python
from csv_inspector import inspect_csv

result = inspect_csv(uploaded_bytes)
```

## Examples

Runnable hosts that show how to embed an agent. They are executable
documentation to read and copy: never built, tagged or published.

| Example | What it shows | How to run |
|---|---|---|
| `csv_inspector_api` | A FastAPI service that embeds `csv-inspector` behind HTTP endpoints (work in progress). | `uv run uvicorn csv_inspector_api.app:create_app --factory` · [README](examples/csv_inspector_api/README.md) |

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

Each agent is self-contained under `agents/<name>/`: its code, tests,
fixtures, docs, demo and tool configuration. Each example is self-contained
under `examples/<name>/` in the same way. The root holds only what the
agents share: the [uv](https://docs.astral.sh/uv/) workspace and lockfile,
the shared ruff defaults, CI and the project's policies. See
[ARCHITECTURE.md](ARCHITECTURE.md) for the full layout and the reasoning
behind it.

## Development

```bash
uv sync --all-packages --all-extras   # every agent (editable) + dev tools, pinned in uv.lock
uv run pre-commit install             # run the CI checks on every commit

uv run ruff check . && uv run ruff format --check .
cd agents/csv_inspector && uv run mypy && uv run pytest --cov
```

The tests are hermetic: no Ollama, no credentials, no network.
[CI](.github/workflows/ci.yml) runs lint, strict type-checking and the tests
for every agent. Tests run on Python 3.10–3.14 (Linux) and on Windows, with
a coverage floor. CI also builds every agent's wheel and sdist, installs
them with pip into clean environments, and smoke-tests them from outside the
repository.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, the code
standards and how to propose changes. Report vulnerabilities privately; see
[SECURITY.md](SECURITY.md).

## License

Released under the [MIT License](LICENSE).
