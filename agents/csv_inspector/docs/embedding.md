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
csv-inspector @ git+https://github.com/deluispablo/data-agent-toolkit@csv-inspector-v0.1.0#subdirectory=agents/csv_inspector
# ...or with the cloud (Gemini) backend and environment loading:
csv-inspector[cloud] @ git+https://github.com/deluispablo/data-agent-toolkit@csv-inspector-v0.1.0#subdirectory=agents/csv_inspector
```

```toml
# pyproject.toml
[project]
dependencies = [
    "csv-inspector[cloud] @ git+https://github.com/deluispablo/data-agent-toolkit@csv-inspector-v0.1.0#subdirectory=agents/csv_inspector",
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
    cloud_model="gemini-2.5-flash",
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
- `Settings` needs no extra package; only the `api` backend needs
  `csv-inspector[cloud]`.

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
Each model may use an equal share of what is left (half for the primary
when there is a fallback), so a hung primary still leaves the fallback
time; time a model does not use carries over. The library enforces it
itself, so custom invokers are bounded too, and also passes each model's
share to the HTTP client (in seconds for Ollama, in milliseconds for
Gemini). When it runs out, `InspectionTimeoutError` is raised.

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

- `INFO`: one line per model attempt and one per success.
- `WARNING`: a failed attempt before the fallback.
- `DEBUG`: backend details.

Records describe sources by path or size (e.g. `<52311 bytes in memory>`)
and never include their content. The Gemini API key never appears in logs
or exception messages.

## 9. Not included, by design

- **An HTTP service or Docker image**: csv-inspector is the component; your
  API is the service.
- **Cloud storage readers** (GCS, S3, Azure): open the object as a binary
  stream with your provider's SDK, or download the bytes, and pass that.
  Seekable readers get the cheapest sampling.
