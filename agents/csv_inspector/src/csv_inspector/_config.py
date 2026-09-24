"""Settings for csv_inspector: explicit by default, environment on request.

:class:`Settings` is a plain, frozen Pydantic model: constructing one never
reads environment variables or files. Hosts that manage their own secrets
build it directly and pass it to :func:`csv_inspector.inspect_csv`.

:func:`load_settings` is the explicit, opt-in way to read settings from the
process environment (and, only if asked, from a ``.env`` file). It needs the
``pydantic-settings`` package from the ``[cloud]`` extra, imported lazily.

:func:`resolve_settings` picks between the two for one call, and
:func:`ensure_backend_ready` checks that a backend is usable as configured,
before any source is read.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, ConfigDict, SecretStr, ValidationError, field_validator

from ._backends import LLMBackend
from ._exceptions import BackendConfigurationError, CredentialsNotConfiguredError

logger = logging.getLogger(__name__)

# Each fallback differs from its primary: a fallback equal to the primary is
# skipped, so it would leave only one model to try out of the box.
DEFAULT_MODEL: str = "qwen2.5-coder:7b"
FALLBACK_MODEL: str = "qwen2.5-coder:3b"
DEFAULT_CLOUD_MODEL: str = "gemini-2.5-flash"
DEFAULT_CLOUD_FALLBACK_MODEL: str = "gemini-2.5-flash-lite"

CLOUD_EXTRA_HINT = "Install it with: pip install 'csv-inspector[cloud]'."


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


class Settings(BaseModel):
    """csv_inspector settings, injectable explicitly or loaded from the environment.

    Constructing ``Settings(...)`` never reads the environment; unknown
    fields are rejected to catch typos. :func:`load_settings` maps each
    field to the upper-cased environment variable of the same name (e.g.
    ``gemini_api_key`` ← ``GEMINI_API_KEY``).

    Attributes:
        llm_backend: Default backend: ``local`` or ``api``.
        ollama_model: Primary local model.
        ollama_fallback_model: Fallback local model.
        gemini_api_key: Gemini Developer API key. Held as a ``SecretStr`` so
            it is masked in ``repr`` and logs.
        google_cloud_project: Vertex AI project.
        google_cloud_location: Vertex AI location.
        cloud_model: Primary cloud model.
        cloud_fallback_model: Fallback cloud model.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    llm_backend: LLMBackend = LLMBackend.LOCAL
    ollama_model: str = DEFAULT_MODEL
    ollama_fallback_model: str = FALLBACK_MODEL
    gemini_api_key: SecretStr | None = None
    google_cloud_project: str | None = None
    google_cloud_location: str | None = None
    cloud_model: str = DEFAULT_CLOUD_MODEL
    cloud_fallback_model: str = DEFAULT_CLOUD_FALLBACK_MODEL

    @field_validator("llm_backend", mode="before")
    @classmethod
    def _normalize_backend(cls, value: object) -> object:
        """Accept the backend case-insensitively (e.g. ``API``, `` local ``)."""
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

    def model_for(self, backend: LLMBackend) -> str:
        """Return the primary model configured for ``backend``."""
        return self.cloud_model if backend is LLMBackend.API else self.ollama_model

    def fallback_model_for(self, backend: LLMBackend) -> str:
        """Return the fallback model configured for ``backend``."""
        if backend is LLMBackend.API:
            return self.cloud_fallback_model
        return self.ollama_fallback_model

    def cloud_credentials(self) -> CloudCredentials:
        """Resolve which cloud credentials to use.

        An API key takes precedence (Gemini Developer API); otherwise both a
        project and a location are required (Vertex AI with Application
        Default Credentials).

        Returns:
            The resolved :class:`CloudCredentials`.

        Raises:
            CredentialsNotConfiguredError: If neither route is fully
                configured. The message names the missing setting(s).
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
            "Default Credentials)."
        )


def load_settings(*, env_file: str | os.PathLike[str] | None = None) -> Settings:
    """Read :class:`Settings` from the process environment, explicitly.

    Only environment variables are read by default. A ``.env`` file is read
    only when ``env_file`` is given: the library never assumes that the
    process's working directory is a safe place to load secrets from.

    Args:
        env_file: Optional path to a ``.env`` file. Environment variables
            take precedence over its values.

    Returns:
        The loaded settings, as a plain :class:`Settings`.

    Raises:
        BackendConfigurationError: If ``pydantic-settings`` (``[cloud]``
            extra) is not installed, or a variable holds an invalid value.
            The message names the variable but never echoes the value,
            which could be a secret.
    """
    try:
        from pydantic_settings import (  # noqa: PLC0415 - optional extra, imported lazily.
            BaseSettings,
            SettingsConfigDict,
        )
    except ImportError as exc:
        raise BackendConfigurationError(
            f"Reading settings from the environment needs 'pydantic-settings'. {CLOUD_EXTRA_HINT}"
        ) from exc

    # Defined here because pydantic-settings is optional. Settings comes first
    # in the bases so its fields, defaults and validators define the schema;
    # it defines no __init__, so BaseSettings.__init__ (which reads the
    # environment and .env sources) is still the one that runs. The config
    # overrides Settings' extra="forbid": unrelated variables in a .env file
    # must be ignored, not rejected.
    class _EnvSettings(Settings, BaseSettings):
        model_config = SettingsConfigDict(frozen=True, extra="ignore")

    try:
        loaded = _EnvSettings(_env_file=None if env_file is None else os.fspath(env_file))
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']).upper()}: {error['msg']}"
            for error in exc.errors()
        )
        raise BackendConfigurationError(f"Invalid csv_inspector settings: {problems}") from None
    # Hand back a plain Settings, not the env-reading subclass: copies and
    # re-validation of the result must never go back to the environment.
    return Settings.model_validate(loaded.model_dump())


def resolve_settings(settings: Settings | None, backend: LLMBackend) -> Settings:
    """Return the settings to use: the injected ones, or the environment's.

    Args:
        settings: Explicitly injected settings. When given, the environment
            is never read.
        backend: The backend the settings are needed for.

    Returns:
        ``settings`` if given; otherwise settings read from the process
        environment (:func:`load_settings`, no ``.env``). On a base install
        without ``pydantic-settings``, the local backend falls back to the
        built-in defaults.

    Raises:
        BackendConfigurationError: If the environment cannot be read for the
            cloud backend (missing extra) or holds an invalid value.
    """
    if settings is not None:
        return settings
    if backend is LLMBackend.LOCAL and importlib.util.find_spec("pydantic_settings") is None:
        logger.debug("pydantic-settings not installed; using built-in local defaults.")
        return Settings()
    return load_settings()


def ensure_backend_ready(backend: LLMBackend, settings: Settings | None = None) -> None:
    """Fail fast if ``backend`` cannot be used, before any source is read.

    The local backend needs nothing up front (Ollama reachability is only
    known when it is called). The cloud backend needs sufficient
    credentials and the ``google-genai`` package.

    Raises:
        BackendConfigurationError: If the ``[cloud]`` extra is missing or a
            setting is invalid.
        CredentialsNotConfiguredError: If the cloud backend has no usable
            credentials.
    """
    if backend is not LLMBackend.API:
        return
    resolve_settings(settings, backend).cloud_credentials()
    try:
        sdk_installed = importlib.util.find_spec("google.genai") is not None
    except ModuleNotFoundError:
        sdk_installed = False
    if not sdk_installed:
        raise BackendConfigurationError(
            f"The 'google-genai' package is required for the 'api' backend. {CLOUD_EXTRA_HINT}"
        )
