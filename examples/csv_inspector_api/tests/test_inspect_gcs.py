"""``POST /inspect/gcs``: a ``gs://`` object sampled with ranged reads, never downloaded."""

from __future__ import annotations

import ast
import logging
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

import csv_inspector_api.app as app_module
import csv_inspector_api.routes.inspect as inspect_route
from csv_inspector_api import create_app
from csv_inspector_api.errors import PROBLEM_MEDIA_TYPE
from csv_inspector_api.settings import ApiSettings
from csv_inspector_api.sources.gcs import (
    GCS_CHUNK_BYTES,
    INSTALL_HINT,
    GcsNotInstalledError,
    chunk_size_for,
    create_client,
    open_gcs_object,
)
from fakes import SAMPLE_CSV, FakeClient, FakeInvoker

SAMPLE = SAMPLE_CSV.read_bytes()
_ROW = "2024-01-15;Cliente;Descripción;1250,50;\n".encode()
BIG = SAMPLE + _ROW * (4 * 1024 * 1024 // len(_ROW))
"""About 4 MiB: the sample's preamble and header, then many data rows."""

URI = "gs://my-bucket/exports/2024/sales.csv"
OBJECT = ("my-bucket", "exports/2024/sales.csv")
SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"


@pytest.fixture
def gcs() -> FakeClient:
    """A fake Cloud Storage client holding :data:`BIG` at :data:`URI`."""
    return FakeClient({OBJECT: BIG})


@pytest.fixture
def app(settings: ApiSettings, invoker: FakeInvoker, gcs: FakeClient) -> FastAPI:
    """The application wired to the fake model and the fake Cloud Storage client."""
    return create_app(settings, model_invoker=invoker, gcs_client=gcs)


@pytest.mark.anyio
async def test_inspects_the_object_with_two_windows_only(
    client: httpx.AsyncClient, gcs: FakeClient, caplog: pytest.LogCaptureFixture
) -> None:
    """Only the head and tail windows are read, then the reader is closed."""
    with caplog.at_level(logging.INFO, logger="csv_inspector_api.inspect"):
        response = await client.post("/inspect/gcs", json={"uri": URI})

    assert response.status_code == 200
    assert response.json()["header_row_index"] == 2
    (blob,) = gcs.blobs
    assert blob.name == OBJECT[1]
    assert blob.open_calls == [("rb", GCS_CHUNK_BYTES)]
    (reader,) = blob.readers
    assert reader.closed
    assert reader.bytes_read <= 4096 + 4096
    assert all(count <= 4096 for _, count in reader.reads)
    assert reader.reads[-1][0] >= len(BIG) - 4096  # the tail window
    assert response.headers["X-Object-Size"] == str(len(BIG))
    assert response.headers["X-Object-Generation"] == str(blob.generation)
    (record,) = caplog.records
    assert repr(URI) in record.getMessage()


@pytest.mark.anyio
async def test_body_is_the_one_of_inspect(client: httpx.AsyncClient, gcs: FakeClient) -> None:
    """Clients get the same body whether the bytes were uploaded or read from GCS."""
    gcs.objects[OBJECT] = SAMPLE

    from_gcs = await client.post("/inspect/gcs", json={"uri": URI})
    uploaded = await client.post("/inspect", files={"file": ("sample.csv", SAMPLE, "text/csv")})

    assert from_gcs.status_code == uploaded.status_code == 200
    assert from_gcs.json() == uploaded.json()


@pytest.mark.anyio
async def test_generation_is_forwarded(client: httpx.AsyncClient, gcs: FakeClient) -> None:
    """A pinned generation reaches the blob and comes back in the header."""
    response = await client.post("/inspect/gcs", json={"uri": URI, "generation": 42})

    assert response.status_code == 200
    (blob,) = gcs.blobs
    assert blob.generation == 42
    assert response.headers["X-Object-Generation"] == "42"


@pytest.mark.anyio
async def test_windows_and_chunk_follow_the_query(
    client: httpx.AsyncClient, gcs: FakeClient
) -> None:
    """``n_bytes`` and ``tail_bytes`` bound what is read; the chunk stays one GCS unit."""
    response = await client.post(
        "/inspect/gcs", params={"n_bytes": 16384, "tail_bytes": 0}, json={"uri": URI}
    )

    assert response.status_code == 200
    (blob,) = gcs.blobs
    assert blob.open_calls == [("rb", GCS_CHUNK_BYTES)]
    assert blob.readers[0].bytes_read <= 16384


@pytest.mark.parametrize(
    ("n_bytes", "tail_bytes", "expected"),
    [
        (4096, 4096, GCS_CHUNK_BYTES),
        (512, 0, GCS_CHUNK_BYTES),
        (0, 0, GCS_CHUNK_BYTES),
        (0, GCS_CHUNK_BYTES, GCS_CHUNK_BYTES),
        (GCS_CHUNK_BYTES + 1, 0, 2 * GCS_CHUNK_BYTES),
    ],
)
def test_chunk_size_rounds_up_to_256_kib(n_bytes: int, tail_bytes: int, expected: int) -> None:
    """The larger window, rounded up to the 256 KiB GCS unit."""
    assert chunk_size_for(n_bytes, tail_bytes) == expected


@pytest.mark.anyio
@pytest.mark.parametrize(
    "body",
    [
        {"uri": "gs://my-bucket"},
        {"uri": "gs://my-bucket/"},
        {"uri": "gs://My-Bucket/file.csv"},
        {"uri": "gs://ab/file.csv"},
        {"uri": "s3://my-bucket/file.csv"},
        {"uri": "https://storage.googleapis.com/my-bucket/file.csv"},
        {"uri": "gs://my-bucket/a\nb.csv"},
        {"uri": URI, "generation": 0},
        {"generation": 1},
    ],
)
async def test_invalid_requests_are_422(
    client: httpx.AsyncClient, gcs: FakeClient, body: dict[str, object]
) -> None:
    """A malformed URI or generation never reaches Cloud Storage."""
    response = await client.post("/inspect/gcs", json=body)

    assert response.status_code == 422
    assert gcs.blobs == []


def test_open_rejects_a_uri_outside_the_pattern() -> None:
    """The source module checks the URI itself, not only the request model."""
    with pytest.raises(ValueError, match="not a gs://bucket/object URI"):
        open_gcs_object("gs://my-bucket", client=FakeClient(), generation=None, chunk_size=1)


@pytest.mark.anyio
async def test_a_failed_read_still_closes_the_reader(
    client: httpx.AsyncClient, gcs: FakeClient
) -> None:
    """The reader is closed in every case; an I/O error is the library's 422."""
    gcs.error = OSError("connection reset")

    response = await client.post("/inspect/gcs", json={"uri": URI})

    assert response.status_code == 422
    assert response.json()["error"] == "FileSampleReadError"
    assert gcs.blobs[0].readers[0].closed


@pytest.mark.anyio
async def test_a_non_seekable_reader_is_refused(
    settings: ApiSettings, invoker: FakeInvoker
) -> None:
    """A forward-only reader would be consumed whole: refused and closed, never sampled."""
    gcs = FakeClient({OBJECT: BIG}, seekable=False)
    app = create_app(settings, model_invoker=invoker, gcs_client=gcs)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/inspect/gcs", json={"uri": URI})

    assert response.status_code == 500
    (reader,) = gcs.blobs[0].readers
    assert reader.closed
    assert reader.reads == []
    assert invoker.calls == []


@pytest.mark.anyio
async def test_cost_raising_overrides_are_refused(
    client: httpx.AsyncClient, gcs: FakeClient
) -> None:
    """The override guard of the other routes applies here too."""
    response = await client.post("/inspect/gcs", params={"backend": "api"}, json={"uri": URI})

    assert response.status_code == 403
    assert gcs.blobs == []


@pytest.fixture
async def unconfigured_client(
    settings: ApiSettings, invoker: FakeInvoker
) -> AsyncIterator[httpx.AsyncClient]:
    """A client for an app without a Cloud Storage client (none built at startup)."""
    app = create_app(settings, model_invoker=invoker)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


@pytest.mark.anyio
async def test_missing_extra_is_a_503_with_the_install_hint(
    unconfigured_client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ``google-cloud-storage`` the route answers 503 and says how to install it."""
    monkeypatch.setitem(sys.modules, "google.cloud.storage", None)

    response = await unconfigured_client.post("/inspect/gcs", json={"uri": URI})

    assert response.status_code == 503
    assert response.headers["content-type"] == PROBLEM_MEDIA_TYPE
    assert response.json()["detail"] == INSTALL_HINT
    assert response.json()["error"] == "GcsNotInstalledError"


@pytest.mark.anyio
async def test_a_client_missing_at_startup_is_built_once_on_demand(
    unconfigured_client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deployment that could not build the client at startup recovers, then reuses it."""
    built: list[str | None] = []

    def _create(project: str | None) -> FakeClient:
        built.append(project)
        return FakeClient({OBJECT: SAMPLE})

    monkeypatch.setattr(inspect_route, "create_client", _create)

    first = await unconfigured_client.post("/inspect/gcs", json={"uri": URI})
    second = await unconfigured_client.post("/inspect/gcs", json={"uri": URI})

    assert first.status_code == second.status_code == 200
    assert built == [None]


@pytest.mark.anyio
async def test_startup_builds_the_client_with_the_configured_project(
    invoker: FakeInvoker, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lifespan builds one client, for ``google_cloud_project`` when set."""
    fake = FakeClient()
    projects: list[str | None] = []

    def _create(project: str | None) -> FakeClient:
        projects.append(project)
        return fake

    monkeypatch.setattr(app_module, "create_client", _create)
    app = create_app(ApiSettings(google_cloud_project="my-project"), model_invoker=invoker)

    async with app.router.lifespan_context(app):
        assert app.state.gcs_client is fake

    assert projects == ["my-project"]


@pytest.mark.anyio
async def test_startup_keeps_an_injected_client(
    app: FastAPI, gcs: FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An injected client is never replaced by a real one."""
    monkeypatch.setattr(app_module, "create_client", pytest.fail)

    async with app.router.lifespan_context(app):
        assert app.state.gcs_client is gcs


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("error", "level"),
    [(GcsNotInstalledError(INSTALL_HINT), logging.INFO), (RuntimeError("no ADC"), logging.WARNING)],
)
async def test_startup_survives_a_client_failure(
    invoker: FakeInvoker,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    error: Exception,
    level: int,
) -> None:
    """Without the extra or credentials the API still starts; the failure is logged."""

    def _fail(project: str | None) -> FakeClient:
        raise error

    monkeypatch.setattr(app_module, "create_client", _fail)
    app = create_app(ApiSettings(), model_invoker=invoker)

    with caplog.at_level(logging.INFO, logger="csv_inspector_api"):
        async with app.router.lifespan_context(app):
            assert app.state.gcs_client is None

    (record,) = [r for r in caplog.records if r.name == "csv_inspector_api.app"]
    assert record.levelno == level
    assert str(error) in record.getMessage()


@pytest.mark.anyio
async def test_openapi_documents_the_object_headers(client: httpx.AsyncClient) -> None:
    """The 200 response declares ``X-Object-Size`` and ``X-Object-Generation``."""
    schema = (await client.get("/openapi.json")).json()

    ok = schema["paths"]["/inspect/gcs"]["post"]["responses"]["200"]
    assert set(ok["headers"]) == {"X-Object-Size", "X-Object-Generation"}


def _google_imports(tree: ast.Module) -> list[tuple[str, bool]]:
    """``(module, at module level)`` of every ``google.*`` import in ``tree``."""
    top_level = {id(node) for node in tree.body}
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        else:
            continue
        found += [(name, id(node) in top_level) for name in names if name.startswith("google")]
    return found


@pytest.mark.parametrize("path", sorted(SOURCE_ROOT.rglob("*.py")), ids=lambda p: p.name)
def test_google_is_imported_lazily_and_never_downloads(path: Path) -> None:
    """No module-level ``google.*`` import, and no whole-object download call."""
    tree = ast.parse(path.read_text(encoding="utf-8"))

    assert [name for name, top in _google_imports(tree) if top] == []
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not {name for name in attributes if name.startswith("download")}


@pytest.mark.parametrize(
    ("project", "kwargs"), [(None, {}), ("my-project", {"project": "my-project"})]
)
def test_create_client_uses_adc_and_the_project_when_set(
    monkeypatch: pytest.MonkeyPatch, project: str | None, kwargs: dict[str, str]
) -> None:
    """``storage.Client`` gets no credentials (ADC) and a project only when configured."""
    calls: list[dict[str, str]] = []

    def _client(**given: str) -> FakeClient:
        calls.append(given)
        return FakeClient()

    monkeypatch.setitem(sys.modules, "google.cloud.storage", SimpleNamespace(Client=_client))

    assert isinstance(create_client(project), FakeClient)
    assert calls == [kwargs]
