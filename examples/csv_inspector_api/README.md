# csv_inspector_api

A small FastAPI service that embeds the
[`csv-inspector`](../../agents/csv_inspector/README.md) agent. It is an
**example**: executable documentation of how to put the agent behind HTTP,
meant to be read and copied. It is never built, tagged or published (see
[Examples](../../ARCHITECTURE.md#examples)).

> **Status: in progress.** `POST /inspect` and `GET /health` work; the demo
> and the full README arrive with the rest of the milestone.

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

### `GET /health`

```bash
curl localhost:8000/health
```

```json
{"status": "ok", "csv_inspector_version": "0.1.0", "api_version": "0.0.0",
 "backend": "local", "model": "qwen2.5-coder:7b", "fallback_model": "qwen2.5-coder:3b"}
```

The health check never contacts Ollama or Gemini: it must stay cheap and
free. It reports the configured backend and models, never an API key, a
cloud project or a location. There is no `?probe=true` reachability check:
it would need `ensure_backend_ready`, which `csv-inspector` does not export
publicly, and the example imports only the public API.

Use `/health` as the liveness and readiness probe (Kubernetes `httpGet`,
Cloud Run startup/liveness probe with `path: /health`).

### `POST /inspect`

Upload the file as `multipart/form-data` in the field `file`:

```bash
curl -F file=@../../agents/csv_inspector/sample.csv "localhost:8000/inspect?timeout_seconds=120"
```

```json
{"encoding": "utf-8", "delimiter": ";", "quotechar": "\"", "escapechar": null,
 "doublequote": true, "header_row_index": 2, "footer_lines": [],
 "columns": [{"name": "Fecha", "inferred_type": "date", "nullable": false,
              "example_values": ["2024-01-15", "2024-01-16"]}, "..."],
 "confidence": 0.9, "notes": "..."}
```

The response is the library's `CSVInspectionResult`, unchanged.

| query parameter | default | bounds | passed to `ainspect_csv` as |
|---|---|---|---|
| `n_bytes` | 4096 | 512–16384 | `n_bytes` |
| `tail_bytes` | 4096 | 0–16384 | `tail_bytes` |
| `timeout_seconds` | `CSV_INSPECTOR_API_DEFAULT_TIMEOUT_SECONDS` | 1–`CSV_INSPECTOR_API_MAX_TIMEOUT_SECONDS` | `timeout_seconds` |

- Only a bounded head and tail of the upload are read and sent to the model.
- An upload larger than `CSV_INSPECTOR_API_MAX_UPLOAD_BYTES` is a `413`.
  The multipart body is already received when the route runs (Starlette
  spools it to a temporary file past 1 MiB), so this limit caps what is
  inspected, not what is received: cap the request size at the reverse
  proxy too.
- The upload's content type is **not** checked: browsers and tools send
  anything from `text/csv` to `application/vnd.ms-excel` or
  `application/octet-stream`, and the library detects what the bytes are.
- Invalid query parameters or a missing `file` field are FastAPI's own
  `422` validation errors (`application/json`); every other error is a
  problem response, see [Errors](#errors).

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
