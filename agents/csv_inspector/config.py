"""Environment-driven settings for the csv_inspector agent.

Settings are read from environment variables and, if present, a ``.env``
file in the current working directory (see ``.env.example`` at the
repository root). This module needs ``pydantic-settings``, which ships in
the optional ``requirements-cloud.txt`` extra, so it is only ever imported
lazily: the default local backend works without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pydantic import SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from backends import LLMBackend
from exceptions import BackendConfigurationError, CredentialsNotConfiguredError
from inspector import DEFAULT_MODEL, FALLBACK_MODEL

DEFAULT_CLOUD_MODEL: str = "gemini-2.5-flash"


class CloudAuthMode(str, Enum):
    """How the cloud backend authenticates.

    Attributes:
        GEMINI_API: Gemini Developer API, authenticated with an API key.
        VERTEX_AI: Vertex AI, authenticated with Application Default
            Credentials for a Google Cloud project and location.
    """

    GEMINI_API = "gemini_api"
    VERTEX_AI = "vertex_ai"


@dataclass(frozen=True)
class CloudCredentials:
    """Resolved credentials for the cloud backend.

    Attributes:
        mode: Which authentication route to use.
        api_key: The Gemini API key, for :attr:`CloudAuthMode.GEMINI_API`.
        project: The Google Cloud project, for :attr:`CloudAuthMode.VERTEX_AI`.
        location: The Google Cloud location, for :attr:`CloudAuthMode.VERTEX_AI`.
    """

    mode: CloudAuthMode
    api_key: SecretStr | None = None
    project: str | None = None
    location: str | None = None

    def describe(self) -> str:
        """Return a log-safe description that never includes the API key."""
        if self.mode is CloudAuthMode.GEMINI_API:
            return "Gemini Developer API (API key)"
        return f"Vertex AI (project={self.project!r}, location={self.location!r}, ADC)"


class Settings(BaseSettings):
    """csv_inspector settings, one field per environment variable.

    Attributes:
        llm_backend: Default backend (``LLM_BACKEND``): ``local`` or ``api``.
        ollama_model: Primary local model (``OLLAMA_MODEL``).
        ollama_fallback_model: Fallback local model (``OLLAMA_FALLBACK_MODEL``).
        gemini_api_key: Gemini Developer API key (``GEMINI_API_KEY``). Held
            as a ``SecretStr`` so it is masked in ``repr`` and logs.
        google_cloud_project: Vertex AI project (``GOOGLE_CLOUD_PROJECT``).
        google_cloud_location: Vertex AI location (``GOOGLE_CLOUD_LOCATION``).
        cloud_model: Primary cloud model (``CLOUD_MODEL``).
        cloud_fallback_model: Fallback cloud model (``CLOUD_FALLBACK_MODEL``).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    llm_backend: LLMBackend = LLMBackend.LOCAL
    ollama_model: str = DEFAULT_MODEL
    ollama_fallback_model: str = FALLBACK_MODEL
    gemini_api_key: SecretStr | None = None
    google_cloud_project: str | None = None
    google_cloud_location: str | None = None
    cloud_model: str = DEFAULT_CLOUD_MODEL
    cloud_fallback_model: str = DEFAULT_CLOUD_MODEL

    @field_validator("llm_backend", mode="before")
    @classmethod
    def _normalize_backend(cls, value: object) -> object:
        """Accept ``LLM_BACKEND`` case-insensitively (e.g. ``API``, `` local ``)."""
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator(
        "gemini_api_key", "google_cloud_project", "google_cloud_location", mode="before"
    )
    @classmethod
    def _blank_as_unset(cls, value: object) -> object:
        """Treat empty values (e.g. ``GEMINI_API_KEY=`` in a ``.env``) as unset."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    def cloud_credentials(self) -> CloudCredentials:
        """Resolve which cloud credentials to use.

        An API key takes precedence (Gemini Developer API); otherwise both a
        project and a location are required (Vertex AI with Application
        Default Credentials).

        Returns:
            The resolved :class:`CloudCredentials`.

        Raises:
            CredentialsNotConfiguredError: If neither route is fully
                configured. The message names the missing variable(s).
        """
        if self.gemini_api_key is not None:
            return CloudCredentials(mode=CloudAuthMode.GEMINI_API, api_key=self.gemini_api_key)
        if self.google_cloud_project and self.google_cloud_location:
            return CloudCredentials(
                mode=CloudAuthMode.VERTEX_AI,
                project=self.google_cloud_project,
                location=self.google_cloud_location,
            )

        if self.google_cloud_project or self.google_cloud_location:
            missing = (
                "GOOGLE_CLOUD_LOCATION" if self.google_cloud_project else "GOOGLE_CLOUD_PROJECT"
            )
            raise CredentialsNotConfiguredError(
                f"Vertex AI needs both GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION; "
                f"{missing} is not set (or set GEMINI_API_KEY to use the Gemini Developer API)."
            )
        raise CredentialsNotConfiguredError(
            "The 'api' backend needs credentials: set GEMINI_API_KEY (Gemini Developer API), "
            "or GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION (Vertex AI with Application "
            "Default Credentials). See .env.example."
        )


def load_settings() -> Settings:
    """Load settings from the environment and an optional ``.env`` file.

    Returns:
        The loaded :class:`Settings`.

    Raises:
        BackendConfigurationError: If a variable holds an invalid value (e.g.
            ``LLM_BACKEND=cloud``). The message names the variable but never
            echoes the value, which could be a secret.
    """
    try:
        return Settings()
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']).upper()}: {error['msg']}"
            for error in exc.errors()
        )
        raise BackendConfigurationError(f"Invalid csv_inspector settings: {problems}") from None
