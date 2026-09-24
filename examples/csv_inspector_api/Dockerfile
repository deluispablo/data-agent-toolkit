# syntax=docker/dockerfile:1
#
# Container image of the csv_inspector_api example, for the cloud (Gemini)
# backend: it has no Ollama. Build it from the repository root, which is the
# uv workspace the example and the agent live in:
#
#   docker build -f examples/csv_inspector_api/Dockerfile -t csv-inspector-api .
#   docker run -p 8000:8000 -e CSV_INSPECTOR_API_LLM_BACKEND=api \
#       -e CSV_INSPECTOR_API_GEMINI_API_KEY=... csv-inspector-api

FROM python:3.14-slim AS build

COPY --from=ghcr.io/astral-sh/uv:0.12.18 /uv /usr/local/bin/uv

# Same venv path in both stages: its scripts refer to it by absolute path.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /repo
COPY . .

# Exactly the locked dependencies of the example, the agent built as a wheel
# (not editable, so the runtime stage needs no source tree), with its [gcs]
# extra (google-cloud-storage) for POST /inspect/gcs. The second sync
# adds the agent's locked [cloud] extra (google-genai) that the api backend
# needs, keeping what the first one installed (--inexact).
RUN uv sync --frozen --no-dev --no-editable --package csv-inspector-api --extra gcs \
 && uv sync --frozen --no-dev --no-editable --package csv-inspector --extra cloud --inexact


FROM python:3.14-slim

RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin app

COPY --from=build /opt/venv /opt/venv
# The example is never built or installed: its package is served from src.
COPY examples/csv_inspector_api/src /app/src

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1

USER app
WORKDIR /app
EXPOSE 8000

CMD ["uvicorn", "csv_inspector_api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
