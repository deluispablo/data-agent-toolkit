# csv_inspector_api

A small FastAPI service that embeds the
[`csv-inspector`](../../agents/csv_inspector/README.md) agent: upload a
CSV or TSV file, get back its encoding, dialect, header row, footer lines
and a preliminary column schema.

## What this is (and is not)

- **An example**: executable documentation of how to put the agent behind
  HTTP, meant to be read and copied. It is small and over-documented on
  purpose.
- **Not a product.** It is never built, tagged or published, has no
  version contract and no `CHANGELOG`: it changes through pull requests
  only (see [Examples](../../ARCHITECTURE.md#examples)). It has no
  authentication, rate limiting or multi-tenancy; add those in your own
  host.
- **Free by default.** Inspections run on a local Ollama. Nothing calls a
  cloud API unless `CSV_INSPECTOR_API_LLM_BACKEND=api` is set.

## Run

From the repository root, after `uv sync --all-packages --all-extras`, with
`ollama serve` running and `qwen2.5-coder:7b` and `qwen2.5-coder:3b` pulled.

**Demo.** Starts the API in-process, waits for `/health`, uploads the
agent's [`sample.csv`](../../agents/csv_inspector/sample.csv), prints the
result and stops:

```bash
uv run examples/csv_inspector_api/main_demo.py
```

Flags mirror the `csv-inspector` CLI: `--file`, `--backend`, `--model`,
`--timeout`, `--log-level`, `--env-file` / `--no-env-file` (default: read
`./.env` if it exists), plus `--port` and `--keep-running` to leave the
server up for `curl` or <http://127.0.0.1:8000/docs>.

**Server.** Serve the application factory with uvicorn:

```bash
cd examples/csv_inspector_api
uv run uvicorn --app-dir src csv_inspector_api.app:create_app --factory
```

Then open <http://127.0.0.1:8000/docs>.

**Configuration.** The API reads only `CSV_INSPECTOR_API_*` environment
variables (all optional). To use a file, copy
[`.env.example`](.env.example) to `.env` and add `--env-file .env` to the
`uvicorn` command.

| variable | default | meaning |
|---|---|---|
| `CSV_INSPECTOR_API_LLM_BACKEND` | `local` | `local` (Ollama) or `api` (Gemini) |
| `CSV_INSPECTOR_API_OLLAMA_MODEL` | `qwen2.5-coder:7b` | primary local model |
| `CSV_INSPECTOR_API_OLLAMA_FALLBACK_MODEL` | `qwen2.5-coder:3b` | fallback local model |
| `CSV_INSPECTOR_API_CLOUD_MODEL` | library default | primary cloud model |
| `CSV_INSPECTOR_API_CLOUD_FALLBACK_MODEL` | library default | fallback cloud model |
| `CSV_INSPECTOR_API_GEMINI_API_KEY` | unset | Gemini Developer API key; never commit it |
| `CSV_INSPECTOR_API_GOOGLE_CLOUD_PROJECT` | unset | Vertex AI project |
| `CSV_INSPECTOR_API_GOOGLE_CLOUD_LOCATION` | unset | Vertex AI location |
| `CSV_INSPECTOR_API_DEFAULT_TIMEOUT_SECONDS` | `60` | time budget of a request |
| `CSV_INSPECTOR_API_MAX_TIMEOUT_SECONDS` | `300` | largest budget a request may ask for |
| `CSV_INSPECTOR_API_MAX_UPLOAD_BYTES` | `268435456` (256 MiB) | larger uploads get `413` |

The `api` backend needs the agent's `[cloud]` extra, which
`uv sync --all-extras` installs.

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
cloud project or a location.

`GET /health?probe=true` also runs the library's `ensure_backend_ready`, the
configuration check every inspection runs first: cloud credentials present,
`google-genai` installed. It makes no network or model call. A failure is a
`503` problem response, for example `CredentialsNotConfiguredError`. It does
**not** prove Ollama is reachable: for the `local` backend it always passes,
because reachability is only known when a model is called. With a custom
`model_invoker` (as in the tests) the check is skipped, as the library skips it.

Use `/health` as the liveness probe and `/health?probe=true` as the readiness
or startup probe (Kubernetes `httpGet`, Cloud Run with `path: /health?probe=true`).

### `POST /inspect`

Upload the file as `multipart/form-data` in the field `file`:

```bash
curl -F file=@../../agents/csv_inspector/sample.csv "localhost:8000/inspect?timeout_seconds=120"
```

Abridged output of a live run with `qwen2.5-coder:7b`:

```json
{"encoding": "utf-8", "delimiter": ";", "quotechar": "\"", "escapechar": "\\",
 "doublequote": true, "header_row_index": 2, "footer_lines": [],
 "columns": [{"name": "Fecha", "inferred_type": "date", "nullable": false,
              "example_values": ["2024-01-15", "2024-01-16", "..."]},
             {"name": "Cliente", "inferred_type": "string", "nullable": false,
              "example_values": ["García, S.L.", "Muñoz Hermanos", "..."]},
             "..."],
 "confidence": 1.0, "notes": null, "footer_rows_to_skip": 0}
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

### `POST /inspect/raw`

Send the file itself as the request body, with or without `Content-Length`
(chunked transfer is fine):

```bash
curl --data-binary @../../agents/csv_inspector/sample.csv \
  -H 'Content-Type: application/octet-stream' "localhost:8000/inspect/raw?timeout_seconds=120"
```

The response, query parameters and errors are those of `POST /inspect`.
The body is streamed to the library as a **non-seekable stream**: it is
read once, as it arrives, and only the head window, the last `tail_bytes`
and one received chunk are held in memory. Nothing is written to disk.

- `Content-Type` is not checked; send `application/octet-stream` or
  `text/csv`, and never `multipart/form-data` (the multipart envelope would
  be inspected as if it were the file).
- `413` when `Content-Length` exceeds `CSV_INSPECTOR_API_MAX_UPLOAD_BYTES`,
  before anything is read, or, without `Content-Length`, as soon as the
  bytes received pass the limit.
- The library reads at most 64 MiB past the head of a non-seekable stream.
  A longer body is not read to its end: it is inspected without a tail, so
  no footer is reported.
- A client that stops sending for `timeout_seconds` fails the read (`422`
  `FileSampleReadError`); a client that disconnects cancels the request and
  releases the worker thread at once.

#### `/inspect` or `/inspect/raw`?

| | `POST /inspect` | `POST /inspect/raw` |
|---|---|---|
| body | `multipart/form-data`, field `file` | the file bytes |
| clients | browsers, HTML forms, `curl -F` | scripts, pipes, proxies, `curl --data-binary` |
| received before inspection | the whole upload, spooled to a temporary file past 1 MiB | nothing: the body is inspected as it streams |
| memory and disk per request | whole upload on disk (or in memory up to 1 MiB) | `n_bytes` + `tail_bytes` + one chunk, no disk |
| tail of a file over 64 MiB | sampled (the file is seekable) | not sampled, no footer |
| 413 | after the upload is received | before reading (`Content-Length`) or while streaming |
| library path | seekable stream: head and tail windows, position restored | non-seekable stream: consumed once |

Use `/inspect` for interactive uploads and for files whose footer matters
past 64 MiB; use `/inspect/raw` when the caller already has a byte stream
and the host should hold as little of it as possible.

**How the body reaches the library.** `ainspect_csv` samples its source in
a worker thread with a blocking `read()`, while Starlette exposes the body
as the async iterator `request.stream()`.
[`AsyncIteratorReader`](src/csv_inspector_api/streaming.py), a small
`io.RawIOBase`, bridges the two: each `read()` schedules the next chunk on
the event loop with `asyncio.run_coroutine_threadsafe` and waits for it.
Reads return at most 8 KiB, so the library's own buffers stay small too.
When the request ends, including by cancellation, the route closes the
reader, which cancels the pending wait and fails the blocked `read()` with
an `OSError`: the worker thread returns instead of waiting for a chunk that
will never come. The wait for one chunk is also bounded by `timeout_seconds`.

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

## How it embeds csv-inspector

The API follows the [embedding guide](../../agents/csv_inspector/docs/embedding.md):

- **Async call on the event loop.** The route awaits `ainspect_csv`, never
  the blocking `inspect_csv` (guide §4).
- **Seekable upload passed as is.** `UploadFile.file` is a spooled temporary
  file: the library samples its head and tail and restores the position.
  No copy, no temporary file of ours (guide §2).
- **Raw body passed as a non-seekable stream.** `/inspect/raw` hands the
  library a blocking reader over the request body, consumed once with
  memory bounded by the sampling windows (guide §2), and releases the
  reader's worker thread when the request ends, since the library cannot
  cancel a blocked read itself (guide §7).
- **Injected settings.** `ApiSettings.to_library_settings()` builds the
  library's `Settings` once per app; the API never calls
  `csv_inspector.load_settings()`, so the library never reads the
  environment (guide §5).
- **One time budget per request.** `timeout_seconds` covers the primary and
  the fallback model together, bounded by `CSV_INSPECTOR_API_MAX_TIMEOUT_SECONDS`
  (guide §6).
- **Exception mapping in one place.** One handler maps every
  `CSVInspectorError` to a status and a problem body; routes catch nothing
  (guide §6, extended in [Errors](#errors)).

## Tests

```bash
uv run --directory examples/csv_inspector_api mypy
uv run --directory examples/csv_inspector_api pytest --cov
```

The tests are hermetic and double as a reference for testing a host that
embeds the agent:

- `create_app(settings, model_invoker=...)` takes a fake async model
  ([`tests/fakes.py`](tests/fakes.py)) that answers, fails, stalls or
  answers garbage. No Ollama, no credentials.
- An `httpx.AsyncClient` over `httpx.ASGITransport` calls the app
  in-process, and a `conftest.py` guard fails any network connection.
- `tests/test_inspect_raw.py` streams a generated 20 MiB body through the
  reader under `tracemalloc` (peak after the head is decoded below twice `n_bytes` + `tail_bytes` +
  64 KiB), and cancels a request mid-body to check that the blocked worker
  thread is released by the reader, not by its timeout.
- `tests/test_errors.py` iterates `csv_inspector.__all__`: a new library
  exception fails the suite until it is mapped to a status.
- `tests/test_embedding_rules.py` checks the source with `ast`: no `print()`
  or `logging.basicConfig()` outside `main_demo.py`, and no import outside
  the library's public API.
- Coverage floor: 90 % (`[tool.coverage.report]` in `pyproject.toml`), enforced in CI.

## Roadmap

- Per-request backend and model override, request-id logging, a Dockerfile.
- `POST /inspect/gcs`: inspect a `gs://` object with ranged reads only.
