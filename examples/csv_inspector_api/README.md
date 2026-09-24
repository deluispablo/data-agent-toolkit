# csv_inspector_api

A small FastAPI service that embeds the
[`csv-inspector`](../../agents/csv_inspector/README.md) agent. It is an
**example**: executable documentation of how to put the agent behind HTTP,
meant to be read and copied. It is never built, tagged or published (see
[Examples](../../ARCHITECTURE.md#examples)).

> **Status: scaffold.** The app boots and serves its OpenAPI schema; the
> endpoints arrive in the next milestone.

## Run

From the repository root, after `uv sync --all-packages --all-extras`:

```bash
cd examples/csv_inspector_api
uv run uvicorn --app-dir src csv_inspector_api.app:create_app --factory
```

Then open <http://127.0.0.1:8000/docs>. Inspections run on a local Ollama
by default (`ollama serve`, with `qwen2.5-coder:7b` and `qwen2.5-coder:3b`
pulled).

## Configuration

The API reads `CSV_INSPECTOR_API_*` environment variables; every one is
listed with its default in [`.env.example`](.env.example). To use a file,
copy it to `.env` and add `--env-file .env` to the `uvicorn` command.
`ApiSettings.to_library_settings()` is the only place that builds the
library's `Settings`: the API never calls `csv_inspector.load_settings()`.

## Endpoints

To be documented with the endpoints (`POST /inspect`, `GET /health`).

## Checks

```bash
uv run --directory examples/csv_inspector_api mypy
uv run --directory examples/csv_inspector_api pytest --cov
```

The tests are hermetic: the app gets a fake model invoker through
`create_app(model_invoker=...)`, and any network connection attempt fails
the test.
