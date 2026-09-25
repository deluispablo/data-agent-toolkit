# csv_inspector_api

A small FastAPI service that embeds the
[`csv-inspector`](../../agents/csv_inspector/README.md) agent: send a CSV
or TSV file and get back its encoding, dialect, header row, footer lines
and column names.

- **An example**: executable documentation of how to put the agent behind
  HTTP. It is never built, tagged or published, and it has no
  `CHANGELOG` (see [Examples](../../ARCHITECTURE.md#examples)). It has no
  authentication, rate limiting or multi-tenancy; add those in your own host.
- **Free by default**: inspections run on a local Ollama. Nothing calls a
  cloud API unless `CSV_INSPECTOR_API_LLM_BACKEND=api` is set, or
  `CSV_INSPECTOR_API_ALLOW_BACKEND_OVERRIDE=true` lets a request ask for it.
- The design notes live next to the code, in the module docstrings; the
  [module map](../../ARCHITECTURE.md#module-map) says which module does what.

## Run

From the repository root, after `uv sync --all-packages --all-extras`, with
`ollama serve` running and `qwen2.5-coder:7b` and `qwen2.5-coder:3b` pulled:

```bash
uv run examples/csv_inspector_api/main_demo.py
```

The demo starts the API in-process, uploads the agent's `sample.csv`,
prints the result and stops. Its flags mirror the `csv-inspector` CLI
(`--file`, `--backend`, `--model`, `--timeout`, `--log-level`,
`--env-file` / `--no-env-file`), plus `--port` and `--keep-running`.

To serve the application factory yourself (then open <http://127.0.0.1:8000/docs>):

```bash
cd examples/csv_inspector_api
uv run uvicorn --app-dir src csv_inspector_api.app:create_app --factory --no-access-log
```

## Configuration

Only `CSV_INSPECTOR_API_*` variables are read, and all of them are
optional. For a file, copy [`.env.example`](.env.example) to `.env` and add
`--env-file .env` to the `uvicorn` command.

| variable | default | meaning |
|---|---|---|
| `LLM_BACKEND` | `local` | `local` (Ollama) or `api` (Gemini) |
| `OLLAMA_MODEL` / `OLLAMA_FALLBACK_MODEL` | `qwen2.5-coder:7b` / `:3b` | local models |
| `OLLAMA_HOST` | SDK default (`http://localhost:11434`) | Ollama server |
| `CLOUD_MODEL` / `CLOUD_FALLBACK_MODEL` | library defaults | cloud models |
| `GEMINI_API_KEY` | unset | Gemini Developer API key; never commit it |
| `GOOGLE_CLOUD_PROJECT` / `GOOGLE_CLOUD_LOCATION` | unset | Vertex AI; the project is also used by the Cloud Storage client |
| `DEFAULT_TIMEOUT_SECONDS` | `60` | budget per request; 90 or more on a CPU-only local deployment |
| `MAX_TIMEOUT_SECONDS` | `300` | largest budget a request may ask for |
| `MAX_UPLOAD_BYTES` | 256 MiB | larger uploads get `413` |
| `ALLOW_BACKEND_OVERRIDE` | `false` | let requests pick the paid backend or cloud models (see below) |

Every name takes the `CSV_INSPECTOR_API_` prefix. The `api` backend needs
the agent's `[cloud]` extra, and `POST /inspect/gcs` needs the example's
`[gcs]` extra; `uv sync --all-extras` installs both.

## Docker

The [`Dockerfile`](Dockerfile) packages the API for the **cloud backend**
(no Ollama in the image; a local backend needs `CSV_INSPECTOR_API_OLLAMA_HOST`).
Build from the repository root:

```bash
docker build -f examples/csv_inspector_api/Dockerfile -t csv-inspector-api .
docker run -p 8000:8000 -e CSV_INSPECTOR_API_LLM_BACKEND=api -e CSV_INSPECTOR_API_GEMINI_API_KEY=... csv-inspector-api
```

The Dockerfile's header comments cover the image (non-root, port 8000),
logging in the container and a Cloud Run deploy command.

## Endpoints

`GET /health` makes no model call and reports no secrets.
`?probe=true` also runs `ensure_backend_ready` (configuration only, `503` on
failure); use it as the readiness probe.

| route | body | source handed to the library | cost of reading the file |
|---|---|---|---|
| `POST /inspect` | `multipart/form-data`, field `file` | seekable spooled upload | 2 bounded reads (the body is first spooled by Starlette) |
| `POST /inspect/raw` | the file bytes (chunked is fine) | non-seekable stream | streams up to 64 MiB past the head in 8 KiB thread hops; no footer beyond that |
| `POST /inspect/gcs` | `{"uri": "gs://bucket/object", "generation": 123}` (`generation` optional) | seekable ranged reader | 1 metadata GET + at most 2 ranged GETs |

Use `/inspect/raw` for pipes and small bodies only: reaching the tail of a
non-seekable stream means reading everything before it.

```bash
curl -F file=@../../agents/csv_inspector/sample.csv "localhost:8000/inspect?timeout_seconds=120"
curl --data-binary @../../agents/csv_inspector/sample.csv -H 'Content-Type: application/octet-stream' "localhost:8000/inspect/raw"
curl -H 'Content-Type: application/json' -d '{"uri": "gs://my-bucket/exports/sales.csv"}' -D - "localhost:8000/inspect/gcs"
```

Every inspection route answers the library's `CSVInspectionResult`
unchanged:

```json
{"encoding": "utf-8", "delimiter": ";", "quotechar": "\"", "escapechar": "\\",
 "doublequote": true, "has_header": true, "header_row_index": 2, "footer_lines": [],
 "columns": ["Fecha", "Cliente", "Importe"], "confidence": 1.0, "footer_rows_to_skip": 0}
```

Every successful inspection also returns `X-Inspection-Model` (the model
whose answer was kept) and, when the backend reports them,
`X-Inspection-Prompt-Tokens` and `X-Inspection-Completion-Tokens` (summed
over the primary and fallback attempts); the body never changes.
`/inspect/gcs` also returns the `X-Object-Size` and `X-Object-Generation`
headers. It reads the object with Application Default Credentials and
needs only `storage.objects.get`. Credentials, IAM, cost and the
unsupported cases are in the [`sources/gcs.py`](src/csv_inspector_api/sources/gcs.py)
docstring. The trade-offs between `/inspect` and `/inspect/raw` (what is
received, when `413` applies, the 64 MiB scan limit) are in the
[`routes/inspect.py`](src/csv_inspector_api/routes/inspect.py) docstring.

| query parameter | default | bounds |
|---|---|---|
| `n_bytes` | 4096 | 512–16384 |
| `tail_bytes` | 4096 | 0–16384 |
| `timeout_seconds` | `DEFAULT_TIMEOUT_SECONDS` | 1–`MAX_TIMEOUT_SECONDS` |
| `backend` | configured | `local`, or `api` when allowed or already configured |
| `model`, `fallback_model` | configured | name token, 1–200 chars; on `api` only when allowed |

> **Cost warning.** `backend=api` sends the sample to Gemini, billed to the
> deployment's credentials, and a model override on a cloud call can pick
> pricier models. Unless the operator sets
> `CSV_INSPECTOR_API_ALLOW_BACKEND_OVERRIDE=true`, both are refused with
> `403` (`BackendOverrideDisabledError`): `backend=api` on a `local`
> deployment, and `model` / `fallback_model` on any call to `api`. The
> default is `false`, and a test checks that nothing in the example turns
> it on. Requests that cannot add cost are always allowed.

## Errors and request ids

Every error is an `application/problem+json` body whose `error` field is
the exception class name, so clients can branch without parsing `detail`:

```json
{"type": "about:blank", "title": "Inspection timed out", "status": 504,
 "detail": "...", "error": "InspectionTimeoutError"}
```

The main statuses are:

- `422`: bad input;
- `504`: timeout;
- `502`: the model failed;
- `503`: the deployment is misconfigured;
- `403` / `413`: raised by the API itself.

The full table, Cloud Storage included, is in the
[`errors.py`](src/csv_inspector_api/errors.py) docstring.

Every response carries `X-Request-ID` (the client's own when it is a sane
token, else a new `uuid4`), and one access line is logged per request. Put
`RequestIdFilter` on your logging **handler**, as `main_demo.py` does, to
see the id on the library's records too. See the
[`request_id.py`](src/csv_inspector_api/request_id.py) docstring.

## How it embeds csv-inspector

It follows the [embedding guide](../../agents/csv_inspector/docs/embedding.md):

- it awaits `ainspect_csv`, never the blocking call (§4);
- it passes each source as it is, never through a temporary file: a
  seekable upload, a streamed body or a Cloud Storage reader, with the
  reader closed when the request ends (§2, §7);
- the library's `Settings` are built once, in
  `ApiSettings.to_library_settings()`, and `load_settings()` is never
  called (§5);
- `timeout_seconds` covers both models within the operator's cap (§6);
- one exception handler maps every `CSVInspectorError` (§6).

## Tests

```bash
uv run --directory examples/csv_inspector_api mypy
uv run --directory examples/csv_inspector_api pytest --cov
```

The tests are hermetic: a fake model, a fake Cloud Storage client, no
network, and a 90 % coverage floor. The
[`tests/conftest.py`](tests/conftest.py) docstring lists what the notable
tests prove.
