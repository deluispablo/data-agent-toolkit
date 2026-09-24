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

## Errors

Every error from `csv-inspector` is answered by one exception handler
([`errors.py`](src/csv_inspector_api/errors.py)) with an RFC 9457-style
body and `Content-Type: application/problem+json`:

```json
{"type": "about:blank", "title": "Inspection timed out", "status": 504,
 "detail": "...", "error": "InspectionTimeoutError"}
```

`error` is the exception class name, so clients can branch on it without
parsing `detail`.

| exception | status | client action |
|---|---|---|
| `EmptySampleError`, `FileSampleReadError` | 422 | fix the input |
| `InspectionTimeoutError` (checked **before** `InspectionFailedError`, its parent) | 504 | retry with a larger `timeout_seconds` or smaller windows |
| `CredentialsNotConfiguredError`, `BackendConfigurationError` | 503 | none: the deployment is misconfigured; retrying elsewhere may help |
| `InspectionFailedError`, `ModelInvocationError`, `ResponseParsingError`, `SchemaValidationError` | 502 | retry, maybe with another model |
| any other `CSVInspectorError` | 500 | report it |

A misconfigured backend is a 503, not a 500: the request was fine and
another instance may be configured correctly. `ValueError` and `TypeError`
from the library are bugs in this host, not domain failures: they are not
handled and surface as FastAPI's plain 500.

## Checks

```bash
uv run --directory examples/csv_inspector_api mypy
uv run --directory examples/csv_inspector_api pytest --cov
```

The tests are hermetic: the app gets a fake model invoker through
`create_app(model_invoker=...)`, and any network connection attempt fails
the test.
