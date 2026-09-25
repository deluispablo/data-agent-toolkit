# Embedding csv-inspector in your application

`csv-inspector` is a library: you add it as a dependency and call it from
your own code, whether that's a web API (FastAPI, Flask, Django), a queue
worker or a data pipeline. This guide covers what a host needs to know:
installation, input sources, sync and async usage, configuration and
secrets, timeouts, error handling, concurrency and logging.

The snippets are illustrative, not a deployable project. Error handling is
kept minimal so the library calls stay visible.

## 1. Add the dependency

Pin a released tag so upgrades are deliberate:

```text
# requirements.txt
csv-inspector @ git+https://github.com/deluispablo/data-agent-toolkit@csv-inspector-v0.4.0#subdirectory=agents/csv_inspector
# ...or with the cloud (Gemini) backend and environment loading:
csv-inspector[cloud] @ git+https://github.com/deluispablo/data-agent-toolkit@csv-inspector-v0.4.0#subdirectory=agents/csv_inspector
```

```toml
# pyproject.toml
[project]
dependencies = [
    "csv-inspector[cloud] @ git+https://github.com/deluispablo/data-agent-toolkit@csv-inspector-v0.4.0#subdirectory=agents/csv_inspector",
]
```

**Only import from `csv_inspector` itself.** Its `__all__` is the stable
API. Modules whose names start with `_`, and anything not exported, are
internal and can change in any release. While the version is `0.x`, minor
releases may still change the public API; the
[CHANGELOG](../CHANGELOG.md) lists every such change.

## 2. Pass the data you already have

`inspect_csv` / `ainspect_csv` take a `CSVSource`. Pick whichever form you
already hold; you never need to write a temporary file.

| You have | Pass | Notes |
|---|---|---|
| A file on disk | `"path/to/file.csv"` or a `Path` | One bounded read per window. A `str` is always a path, never CSV content. |
| Bytes in memory (a small upload, a downloaded blob) | `bytes` / `bytearray` / `memoryview` | Only the sampled windows are copied. |
| A seekable binary stream (an open file, `UploadFile.file`, `io.BytesIO`, a cloud SDK reader opened with `"rb"`) | the stream | Sampled from its **current position**; the position is **restored** afterwards, so you can still read it yourself. |
| A non-seekable stream (a raw request body) | the stream | **Consumed** in a single pass; memory stays bounded by `n_bytes + tail_bytes`. You cannot re-read it afterwards. At most 64 MiB are read past the head; a longer stream gets no tail sample, so no footer is reported. |

Text-mode streams (`open(path)` without `"b"`, `io.StringIO`) are rejected
with `TypeError`: pass bytes so the library can detect the encoding itself.

The windows bound what is read; the prompt then keeps only the first 15
lines of the head and the last 10 of the tail (a file of up to 25 lines is
sent whole). These line bounds are fixed: a larger `n_bytes` helps only
when lines are longer than the window, such as a file with hundreds of
columns.

If your loader never has footers, pass `tail_bytes=0` (`--tail-bytes 0`
on the CLI, `?tail_bytes=0` on the example API): one read instead of two,
no tail tokens in the prompt, and `footer_lines` is always `[]`. On a file
larger than the head window, `covers_whole_file` is then `False`: the end
was never seen, so no footer can be reported. The head keeps its line
bound. Measured cost and accuracy of this mode are in
[evaluation.md](evaluation.md#cost-levers).

If your host lets callers choose the sample windows, validate them against
the library's own limits instead of copying the numbers, so an upgrade can
never desynchronize the two. The same goes for a custom model seam: type it
with the exported invoker aliases.

```python
from csv_inspector import (
    DEFAULT_SAMPLE_BYTES,  # head window by default (4 KiB)
    DEFAULT_TAIL_BYTES,  # tail window by default (4 KiB)
    MAX_SAMPLE_BYTES,  # upper bound of each window (16 KiB)
    AsyncModelInvoker,  # async (prompt, model) -> raw text
    ModelInvoker,  # (prompt, model) -> raw text
)


def checked_window(value: int | None, default: int) -> int:
    value = default if value is None else value
    if not 0 <= value <= MAX_SAMPLE_BYTES:
        raise ValueError(f"window must be 0..{MAX_SAMPLE_BYTES} bytes")
    return value
```

## 3. Synchronous hosts (Flask, Django, workers)

```python
from flask import Flask, jsonify, request

from csv_inspector import CSVInspectorError, Settings, inspect_csv

app = Flask(__name__)
SETTINGS = Settings()  # built once from your own config; see section 5


@app.post("/inspect")
def inspect_upload():
    upload = request.files["file"]  # a seekable, spooled temporary file
    try:
        result = inspect_csv(upload.stream, settings=SETTINGS, timeout_seconds=30)
    except CSVInspectorError as exc:
        return jsonify(error=str(exc)), status_for(exc)  # see section 6
    return jsonify(result.model_dump())
```

`inspect_csv` blocks the calling thread for up to `timeout_seconds` while
the model answers, which is normal for a threaded WSGI server or a worker.

## 4. Asynchronous hosts (FastAPI, Starlette, aiohttp)

**Never call `inspect_csv` directly on the event loop.** It blocks, and
every other request on that worker stalls until the model answers. Use
`ainspect_csv`:

```python
from fastapi import FastAPI, HTTPException, UploadFile

from csv_inspector import CSVInspectorError, Settings, ainspect_csv

app = FastAPI()
SETTINGS = Settings()


@app.post("/inspect")
async def inspect_upload(file: UploadFile):
    try:
        result = await ainspect_csv(file.file, settings=SETTINGS, timeout_seconds=30)
    except CSVInspectorError as exc:
        raise HTTPException(status_for(exc), str(exc)) from exc
    return result.model_dump()
```

`ainspect_csv` reads the source in a worker thread (`asyncio.to_thread`),
calls the model through the SDK's native async client, and enforces the
time budget with `asyncio.wait_for`, which cancels the pending request.

If some code path has to stay synchronous, offload it explicitly:

```python
result = await asyncio.to_thread(inspect_csv, data, settings=SETTINGS, timeout_seconds=30)
```

## 5. Configuration and secrets

Hosts should **inject** configuration:

```python
from pydantic import SecretStr

from csv_inspector import LLMBackend, Settings, inspect_csv

settings = Settings(
    gemini_api_key=SecretStr(secret_manager.get("gemini-api-key")),
    cloud_model="gemini-3.6-flash",
)
result = inspect_csv(data, backend=LLMBackend.API, settings=settings)
```

- **An explicit `Settings` never reads the environment.** Not the process
  variables, not a `.env` file, and the environment loader is not even
  called. That makes it safe for multi-tenant hosts: each tenant can pass
  its own `Settings`, concurrently, with no cross-talk, and tests can't be
  affected by the machine they run on.
- `Settings` is frozen and rejects unknown fields, so a typo such as
  `gemini_key=` fails immediately.
- `Settings` and `load_settings` need no extra package; only the `api`
  backend needs `csv-inspector[cloud]`.
- The Ollama server is part of `Settings` too: pass
  `Settings(ollama_host="http://ollama:11434")` instead of exporting
  `OLLAMA_HOST`. Left unset, the Ollama SDK uses its default (and still
  honours `OLLAMA_HOST` in the process environment).

For standalone use you can omit `settings`: the library then reads the
documented environment variables (`LLM_BACKEND`, `OLLAMA_MODEL`,
`GEMINI_API_KEY`, ...), but **never a `.env` file**. The library does not
assume your process's working directory is a safe place to load secrets
from. To use a `.env` file, ask for it explicitly:

```python
from csv_inspector import load_settings

settings = load_settings(env_file="/etc/myapp/csv-inspector.env")  # variables still win
```

Configuration problems (a missing credential, a missing SDK, an invalid
value) raise `BackendConfigurationError` **before the source is read**, so
a non-seekable request body is never consumed only to fail on
configuration.

The same check is public, so a host can run it before the first request,
at startup or in a readiness probe. It makes no network or model call:

```python
from csv_inspector import BackendConfigurationError, LLMBackend, ensure_backend_ready

try:
    ensure_backend_ready(LLMBackend.API, settings)
except BackendConfigurationError as exc:  # CredentialsNotConfiguredError included
    ...  # report "not ready"
```

For the `local` backend it always passes: whether Ollama is reachable is
only known when it is called.

## 6. Timeouts and error handling

`timeout_seconds` is **one budget for the whole model phase**, shared by
the primary and fallback models, so a request never waits longer than it.
With a fallback, the primary may use about 70 % of the budget and the
fallback everything left, so a hung primary still leaves the fallback
about 30 %; time the primary does not use carries over. On a CPU-only
local deployment, budget for a cold model load: 90 s or more. The library enforces it
itself, so custom invokers are bounded too, and also passes each model's
share to the HTTP client (in seconds for Ollama, in milliseconds for
Gemini). When it runs out, `InspectionTimeoutError` is raised.

A custom invoker returns the model's raw text, parsed and grounded
exactly like a built-in backend's answer. When it calls another model
with the library's prompt, it gets back what the prompt asks for: a JSON
object with the dialect, `has_header`, `header_row_index`, `columns`,
`confidence` and `footer_first_line`, the first non-blank line after the
data (or `null`); the library then reads the whole footer from the file
and returns it as `footer_lines`. A hard-coded answer in the older shape,
with a `footer_lines` list instead, is still accepted: its first
non-blank line is used as `footer_first_line`.

A custom invoker may raise any exception: apart from
`BackendConfigurationError`, which is re-raised at once, it counts as a
failed attempt, the fallback model is tried, and if every model fails the
original exceptions are in `InspectionFailedError.attempts`.

A reasonable mapping to HTTP status codes:

```python
from csv_inspector import (
    BackendConfigurationError,
    CSVInspectorError,
    EmptySampleError,
    FileSampleReadError,
    InspectionTimeoutError,
)


def status_for(exc: CSVInspectorError) -> int:
    if isinstance(exc, (EmptySampleError, FileSampleReadError)):
        return 422  # the client's input
    if isinstance(exc, InspectionTimeoutError):
        return 504  # before InspectionFailedError: it is a subclass
    if isinstance(exc, BackendConfigurationError):
        return 500  # our deployment: missing credentials or SDK
    return 502  # the model backend failed or answered nonsense
```

`ValueError` (a byte budget or timeout out of range) and `TypeError` (an
unsupported source) are programming errors in the host, not domain
failures; let them surface.

## 7. Concurrency and thread-safety

- The library keeps **no mutable module-level state**, and **every call
  creates, and closes, its own model client**. Calls from many threads or
  many tasks at once are independent. The test suite checks this with
  parallel calls and per-tenant settings.
- With `timeout_seconds` on the **sync** API, the model call runs in a
  short-lived daemon worker thread. When the budget runs out, your call
  returns on time, but Python cannot kill a blocking call, so that thread
  finishes in the background (being a daemon, it never keeps the
  interpreter from exiting). For the built-in backends this is brief, because the
  same timeout is set on their HTTP client. A custom sync invoker should
  honour its own timeout too.
- `timeout_seconds` does not cover **sampling**. Reading a non-seekable
  stream is bounded in bytes (at most 64 MiB past the head), not in time:
  a stream that stalls blocks in `read()`, and in `ainspect_csv` that
  happens in a worker thread that cannot be cancelled. Set a read timeout
  on the stream itself (e.g. the socket or HTTP client timeout).
- **Limiting concurrency is the host's job.** A local Ollama processes a
  few requests at a time; bound concurrent inspections with your server's
  worker settings or an `asyncio.Semaphore`.

Without a bound, a burst of requests becomes a burst of model calls: on the
cloud backend it spends the per-minute quota at once (and every `429` it
earns is retried, costing another full call), and on a local one it queues
every request on the same GPU until they time out together. In an async
host, create one `asyncio.Semaphore(n)` at startup (in the lifespan, so it
belongs to the serving event loop), acquire it around each `ainspect_csv`
call with `asyncio.wait_for(semaphore.acquire(), timeout=...)`, release it
in a `finally`, and answer a request that waited too long with `503` and
`Retry-After` instead of letting it queue forever. Acquire before reading a
streamed body, so a waiting request holds no memory, and never gate the
health check. Size `n` to what the backend serves in parallel; it is per
process, so the total is `n` times the number of workers. A semaphore bounds
calls in flight, not calls per day: on the free tier, count calls too (see
"Running on the free tier" in the README). The
[example API](https://github.com/deluispablo/data-agent-toolkit/tree/main/examples/csv_inspector_api)
does exactly this (`CSV_INSPECTOR_API_MAX_CONCURRENT_INSPECTIONS` and
`CSV_INSPECTOR_API_QUEUE_TIMEOUT_SECONDS`, in `routes/inspect.py`).

- Injecting a shared, pre-built HTTP client for very high throughput is
  deliberately not supported yet.

## 8. Logging

The library logs through the standard `logging` module under the
`csv_inspector` logger. It attaches only a `NullHandler`, never calls
`logging.basicConfig()` and never prints, so its records go wherever your
application routes them:

```python
import logging

logging.getLogger("csv_inspector").setLevel(logging.WARNING)
```

- `INFO`: one line per model attempt, and two per success: the outcome and
  its usage (`model=… prompt_tokens=… completion_tokens=… latency=…s
  attempts=… retries=…`).
- `WARNING`: a failed attempt before the fallback.
- `DEBUG`: backend details.

To meter cost per request, log or export `result.usage` yourself (it is
not in the serialized result); the API example returns the model and token
counts as `X-Inspection-*` response headers. Its type, `Usage`, is public,
so host code can annotate it:

```python
from csv_inspector import Usage, inspect_csv


def record_usage(usage: Usage | None) -> None:
    if usage is not None:
        metrics.record(usage.model, usage.prompt_tokens, usage.completion_tokens)


record_usage(inspect_csv(path).usage)
```

Records describe sources by path or size (e.g. `<52311 bytes in memory>`)
and never include their content. The Gemini API key never appears in logs
or exception messages.

## 9. Not included, by design

- **An HTTP service or Docker image**: csv-inspector is the component; your
  API is the service.
- **Cloud storage readers** (GCS, S3, Azure): open the object as a binary
  stream with your provider's SDK, or download the bytes, and pass that.
  Seekable readers get the cheapest sampling.
