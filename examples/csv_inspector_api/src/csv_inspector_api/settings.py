"""Configuration of the API, read from ``CSV_INSPECTOR_API_*`` environment variables."""

from __future__ import annotations

from typing import Any

from csv_inspector import LLMBackend, Settings
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ``Settings()`` never reads the environment: it only carries the library defaults.
_LIBRARY_DEFAULTS = Settings()

_MIB = 1024 * 1024


class ApiSettings(BaseSettings):
    """Settings of the API host.

    The host owns configuration: these settings are read from the
    environment here, and :meth:`to_library_settings` is the one place that
    turns them into the library's :class:`csv_inspector.Settings`. The API
    never calls :func:`csv_inspector.load_settings`.

    Attributes:
        llm_backend: Backend used for inspections: ``local`` (Ollama) or ``api``.
        ollama_model: Primary local model.
        ollama_fallback_model: Fallback local model.
        ollama_host: Base URL of the Ollama server; ``None`` keeps the SDK default.
        cloud_model: Primary cloud model; ``None`` keeps the library default.
        cloud_fallback_model: Fallback cloud model; ``None`` keeps the library default.
        gemini_api_key: Gemini Developer API key, for the cloud backend only.
        google_cloud_project: Vertex AI project, for the cloud backend, and project
            of the Cloud Storage client of ``POST /inspect/gcs``.
        google_cloud_location: Vertex AI location, for the cloud backend only.
        default_timeout_seconds: Time budget of one inspection request.
        max_timeout_seconds: Largest time budget a request may ask for.
        max_upload_bytes: Largest accepted upload; larger ones are rejected early.
        allow_backend_override: Whether a request may move a local deployment to
            ``backend=api`` (the paid cloud backend) or choose the models of a
            cloud call. ``False`` answers such requests with 403.
    """

    model_config = SettingsConfigDict(env_prefix="CSV_INSPECTOR_API_", frozen=True)

    llm_backend: LLMBackend = _LIBRARY_DEFAULTS.llm_backend
    ollama_model: str = _LIBRARY_DEFAULTS.ollama_model
    ollama_fallback_model: str = _LIBRARY_DEFAULTS.ollama_fallback_model
    ollama_host: str | None = None
    cloud_model: str | None = None
    cloud_fallback_model: str | None = None
    gemini_api_key: SecretStr | None = None
    google_cloud_project: str | None = None
    google_cloud_location: str | None = None
    default_timeout_seconds: float = Field(default=60, gt=0)
    max_timeout_seconds: float = Field(default=300, gt=0)
    max_upload_bytes: int = Field(default=256 * _MIB, gt=0)
    # Cloud calls cost money: a caller may only switch this deployment to the
    # paid backend, or pick its (pricier) models, when the operator opts in.
    # Keep the default False.
    allow_backend_override: bool = False

    @model_validator(mode="after")
    def _check_timeouts(self) -> ApiSettings:
        """Reject a default time budget above the cap.

        Returns:
            The validated settings.

        Raises:
            ValueError: If ``default_timeout_seconds`` exceeds ``max_timeout_seconds``.
        """
        if self.default_timeout_seconds > self.max_timeout_seconds:
            msg = "default_timeout_seconds must not exceed max_timeout_seconds"
            raise ValueError(msg)
        return self

    def to_library_settings(self) -> Settings:
        """Build the library settings that every inspection uses.

        Cloud fields left unset keep the library defaults.

        Returns:
            The ``csv_inspector`` settings matching this configuration.
        """
        values: dict[str, Any] = {
            "llm_backend": self.llm_backend,
            "ollama_model": self.ollama_model,
            "ollama_fallback_model": self.ollama_fallback_model,
            "ollama_host": self.ollama_host,
            "cloud_model": self.cloud_model,
            "cloud_fallback_model": self.cloud_fallback_model,
            "gemini_api_key": self.gemini_api_key,
            "google_cloud_project": self.google_cloud_project,
            "google_cloud_location": self.google_cloud_location,
        }
        return Settings(**{name: value for name, value in values.items() if value is not None})
