"""``POST /inspect``: upload in, inspection out, library errors mapped."""

from __future__ import annotations

import logging
from typing import Any

import csv_inspector
import httpx
import pytest
from fastapi import FastAPI
from pydantic import SecretStr

import csv_inspector_api.routes.inspect as inspect_route
from csv_inspector_api import create_app
from csv_inspector_api.errors import PROBLEM_MEDIA_TYPE, problem_responses
from csv_inspector_api.settings import ApiSettings
from fakes import SAMPLE_CSV, FakeInvoker

SAMPLE = SAMPLE_CSV.read_bytes()


def _upload(data: bytes = SAMPLE, content_type: str = "text/csv") -> dict[str, Any]:
    """The ``files=`` argument of a multipart upload in the ``file`` field."""
    return {"file": ("sample.csv", data, content_type)}


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record the keyword arguments the route passes to ``ainspect_csv``."""
    calls: list[dict[str, Any]] = []
    real = csv_inspector.ainspect_csv

    async def _spy(source: Any, /, **kwargs: Any) -> csv_inspector.CSVInspectionResult:
        calls.append({"source": source, **kwargs})
        return await real(source, **kwargs)

    monkeypatch.setattr(inspect_route, "ainspect_csv", _spy)
    return calls


@pytest.mark.anyio
async def test_inspects_the_sample(
    client: httpx.AsyncClient, invoker: FakeInvoker, caplog: pytest.LogCaptureFixture
) -> None:
    """The agent's sample comes back inspected, and one INFO line is logged."""
    with caplog.at_level(logging.INFO, logger="csv_inspector_api.inspect"):
        response = await client.post("/inspect", files=_upload())

    assert response.status_code == 200
    body = response.json()
    assert body["delimiter"] == ";"
    assert body["header_row_index"] == 2
    assert [c["name"] for c in body["columns"]] == [
        "Fecha",
        "Cliente",
        "Descripción",
        "Importe",
        "Observaciones",
    ]
    assert body == csv_inspector.CSVInspectionResult.model_validate(body).model_dump(mode="json")
    assert [model for _, model in invoker.calls] == ["qwen2.5-coder:7b"]
    (record,) = caplog.records
    assert "'sample.csv'" in record.getMessage()
    assert "local/qwen2.5-coder:7b" in record.getMessage()
    assert "confidence 0.90" in record.getMessage()


@pytest.mark.anyio
async def test_passes_the_upload_and_defaults_to_the_library(
    client: httpx.AsyncClient, settings: ApiSettings, spy: list[dict[str, Any]]
) -> None:
    """The seekable upload is passed as is, with the default windows and time budget."""
    response = await client.post("/inspect", files=_upload())

    assert response.status_code == 200
    (call,) = spy
    assert hasattr(call["source"], "seek")
    assert not isinstance(call["source"], bytes | bytearray)
    assert call["n_bytes"] == 4096
    assert call["tail_bytes"] == 4096
    assert call["timeout_seconds"] == settings.default_timeout_seconds
    assert call["backend"] is csv_inspector.LLMBackend.LOCAL
    assert call["settings"] == settings.to_library_settings()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "params",
    [
        {"n_bytes": 512},
        {"n_bytes": 16384},
        {"tail_bytes": 0},
        {"tail_bytes": 16384},
        {"timeout_seconds": 1},
        {"timeout_seconds": 5},
    ],
)
async def test_query_bounds_are_inclusive(
    client: httpx.AsyncClient, spy: list[dict[str, Any]], params: dict[str, float]
) -> None:
    """Values on the bounds are accepted and reach the library."""
    response = await client.post("/inspect", params=params, files=_upload())

    assert response.status_code == 200
    ((name, value),) = params.items()
    assert spy[0][name] == value


@pytest.mark.anyio
@pytest.mark.parametrize(
    "params",
    [
        {"n_bytes": 511},
        {"n_bytes": 16385},
        {"tail_bytes": -1},
        {"tail_bytes": 16385},
        {"timeout_seconds": 0.5},
        {"timeout_seconds": 5.5},  # above settings.max_timeout_seconds
        {"n_bytes": "many"},
    ],
)
async def test_query_out_of_bounds_is_rejected(
    client: httpx.AsyncClient, invoker: FakeInvoker, params: dict[str, float | str]
) -> None:
    """Values outside the bounds are FastAPI 422s and never reach the model."""
    response = await client.post("/inspect", params=params, files=_upload())

    assert response.status_code == 422
    assert response.headers["content-type"] == "application/json"
    assert invoker.calls == []


@pytest.mark.anyio
async def test_missing_file_field_is_rejected(
    client: httpx.AsyncClient, invoker: FakeInvoker
) -> None:
    """A request without the ``file`` field is a FastAPI 422."""
    response = await client.post("/inspect", files={"other": ("x.csv", SAMPLE, "text/csv")})

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "file"]
    assert invoker.calls == []


@pytest.mark.anyio
async def test_content_type_is_not_enforced(client: httpx.AsyncClient) -> None:
    """Whatever type the client declares, the bytes decide."""
    response = await client.post("/inspect", files=_upload(content_type="application/vnd.ms-excel"))

    assert response.status_code == 200


@pytest.mark.anyio
async def test_oversize_upload_is_rejected(invoker: FakeInvoker) -> None:
    """An upload over max_upload_bytes is a 413 problem; the model is never called."""
    app = create_app(ApiSettings(max_upload_bytes=len(SAMPLE) - 1), model_invoker=invoker)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/inspect", files=_upload())

    assert response.status_code == 413
    assert response.headers["content-type"] == PROBLEM_MEDIA_TYPE
    body = response.json()
    assert body["error"] == "UploadTooLargeError"
    assert str(len(SAMPLE)) in body["detail"]
    assert invoker.calls == []


@pytest.mark.anyio
async def test_upload_at_the_limit_is_accepted(invoker: FakeInvoker) -> None:
    """An upload of exactly max_upload_bytes is inspected."""
    app = create_app(ApiSettings(max_upload_bytes=len(SAMPLE)), model_invoker=invoker)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/inspect", files=_upload())

    assert response.status_code == 200


@pytest.mark.anyio
async def test_empty_file_is_422(client: httpx.AsyncClient, invoker: FakeInvoker) -> None:
    """An empty upload is the client's input problem; the model is never called."""
    response = await client.post("/inspect", files=_upload(b""))

    assert response.status_code == 422
    assert response.headers["content-type"] == PROBLEM_MEDIA_TYPE
    assert response.json()["error"] == "EmptySampleError"
    assert invoker.calls == []


@pytest.mark.anyio
async def test_timeout_is_504(client: httpx.AsyncClient, invoker: FakeInvoker) -> None:
    """A model slower than the time budget is a 504."""
    invoker.delay = 30

    response = await client.post("/inspect", params={"timeout_seconds": 1}, files=_upload())

    assert response.status_code == 504
    assert response.json()["error"] == "InspectionTimeoutError"


@pytest.mark.anyio
async def test_backend_configuration_error_is_503(
    client: httpx.AsyncClient, invoker: FakeInvoker
) -> None:
    """A misconfigured backend is a 503, and the fallback model is not tried."""
    invoker.error = csv_inspector.BackendConfigurationError("SDK not installed")

    response = await client.post("/inspect", files=_upload())

    assert response.status_code == 503
    assert response.json()["error"] == "BackendConfigurationError"
    assert len(invoker.calls) == 1


@pytest.mark.anyio
async def test_model_failure_is_502(client: httpx.AsyncClient, invoker: FakeInvoker) -> None:
    """When primary and fallback models both fail, the answer is a 502."""
    invoker.error = ConnectionError("model server went away")

    response = await client.post("/inspect", files=_upload())

    assert response.status_code == 502
    assert response.json()["error"] == "InspectionFailedError"
    assert [model for _, model in invoker.calls] == ["qwen2.5-coder:7b", "qwen2.5-coder:3b"]


@pytest.mark.anyio
async def test_garbage_answer_is_502(client: httpx.AsyncClient, invoker: FakeInvoker) -> None:
    """A model answer that is not JSON goes through ResponseParsingError to a 502."""
    invoker.answer = "I think it is a CSV file."

    response = await client.post("/inspect", files=_upload())

    assert response.status_code == 502
    assert response.json()["error"] == "InspectionFailedError"


@pytest.mark.anyio
async def test_secrets_do_not_leak_into_error_bodies_or_logs(
    invoker: FakeInvoker, caplog: pytest.LogCaptureFixture
) -> None:
    """A configured key in a model error reaches neither the response body nor any log.

    The library redacts the configured key from its own log lines, whatever
    the invoker; the API adds nothing that could carry it.
    """
    secret = "AIza-test-secret-value"
    settings = ApiSettings(gemini_api_key=SecretStr(secret))
    app: FastAPI = create_app(settings, model_invoker=invoker)
    invoker.error = PermissionError(f"API key {secret} was rejected")
    transport = httpx.ASGITransport(app=app)

    with caplog.at_level(logging.DEBUG):
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post("/inspect", files=_upload())

    assert response.status_code == 502
    assert secret not in response.text
    assert any(r.name.startswith("csv_inspector_api") for r in caplog.records)
    assert "***" in caplog.text
    assert secret not in caplog.text


@pytest.mark.anyio
async def test_openapi_documents_the_contract(client: httpx.AsyncClient) -> None:
    """The schema shows the library result, the problem responses and the real bounds."""
    operation = (await client.get("/openapi.json")).json()["paths"]["/inspect"]["post"]

    assert operation["tags"] == ["inspection"]
    ok = operation["responses"]["200"]["content"]["application/json"]
    assert ok["schema"] == {"$ref": "#/components/schemas/CSVInspectionResult"}
    csv_inspector.CSVInspectionResult.model_validate(ok["example"])
    for status in ("413", "422", "502", "503", "504"):
        content = operation["responses"][status]["content"]
        assert content[PROBLEM_MEDIA_TYPE]["schema"]["title"] == "ProblemDetails"
    assert "application/json" in operation["responses"]["422"]["content"]
    timeout = next(p for p in operation["parameters"] if p["name"] == "timeout_seconds")
    assert timeout["schema"]["maximum"] == 5
    assert timeout["schema"]["default"] == 2


def test_problem_responses_without_422() -> None:
    """Statuses other than 422 only document the problem body."""
    assert list(problem_responses(504)[504]["content"]) == [PROBLEM_MEDIA_TYPE]
