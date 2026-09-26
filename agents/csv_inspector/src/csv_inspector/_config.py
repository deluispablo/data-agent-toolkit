"""Settings for csv_inspector: explicit by default, environment on request.

Constructing :class:`Settings` never reads the environment or a file;
:func:`load_settings` is the opt-in reader (no extra package needed),
:func:`resolve_settings` picks one of the two for a call, and
:func:`ensure_backend_ready` checks a backend before any source is read.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, SecretStr, ValidationError, field_validator

from ._backends import LLMBackend
from ._exceptions import BackendConfigurationError, CredentialsNotConfiguredError

logger = logging.getLogger(__name__)

# The local defaults follow the M8 model comparison (docs/evaluation.md,
# "Model comparison 2026-09"): the fallback is the next-smallest model that
# qualifies, or the primary again when none does, as for qwen2.5-coder:7b. A
# fallback equal to the primary is skipped, so the local backend tries one
# model out of the box and gives it the whole time budget. The cloud
# fallback differs from its primary.
DEFAULT_MODEL: str = "qwen2.5-coder:7b"
FALLBACK_MODEL: str = "qwen2.5-coder:7b"
DEFAULT_CLOUD_MODEL: str = "gemini-3.6-flash"
DEFAULT_CLOUD_FALLBACK_MODEL: str = "gemini-flash-lite-latest"

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

    Construction never reads the environment and rejects unknown fields;
    :func:`load_settings` reads each field from its upper-cased name.

    Attributes:
        llm_backend: Default backend: ``local`` or ``api``.
        ollama_model: Primary local model.
        ollama_fallback_model: Fallback local model.
        ollama_host: Base URL of the Ollama server; ``None`` is the SDK's
            default (or ``OLLAMA_HOST``).
        gemini_api_key: Gemini Developer API key, masked in ``repr`` and logs.
        google_cloud_project: Vertex AI project.
        google_cloud_location: Vertex AI location.
        cloud_model: Primary cloud model.
        cloud_fallback_model: Fallback cloud model.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    llm_backend: LLMBackend = LLMBackend.LOCAL
    ollama_model: str = DEFAULT_MODEL
    ollama_fallback_model: str = FALLBACK_MODEL
    ollama_host: str | None = None
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
        "ollama_host",
        "gemini_api_key",
        "google_cloud_project",
        "google_cloud_location",
        mode="before",
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
        """The cloud credentials: an API key first, else a Vertex AI project and location.

        Raises:
            CredentialsNotConfiguredError: If neither route is fully
                configured; the message names the missing setting(s).
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


# The environment variable of each Settings field: its name, upper-cased.
_ENV_VARIABLES: Mapping[str, str] = MappingProxyType(
    {name.upper(): name for name in Settings.model_fields}
)


def _unquote(value: str) -> str:
    """Strip matching surrounding quotes, or a trailing `` # comment`` from a bare value."""
    for quote in "\"'":
        if len(value) > 1 and value.startswith(quote) and value.endswith(quote):
            return value[1:-1]
    return value.split(" #", 1)[0].rstrip()


def _read_env_file(path: str | os.PathLike[str]) -> dict[str, str]:
    """Read ``KEY=VALUE`` lines from a ``.env`` file; a missing file reads as empty.

    Blank and ``#`` lines, ``export``, quoted values and a trailing `` # comment``
    are understood, keys case-insensitively; not interpolation, escapes or
    multi-line values.

    Raises:
        BackendConfigurationError: If the file exists but cannot be read.
    """
    file = Path(path)
    if not file.is_file():
        return {}
    try:
        text = file.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise BackendConfigurationError(f"Cannot read settings file '{file}': {exc}") from None
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, separator, value = line.partition("=")
        if not separator or line.startswith("#"):
            continue
        values[key.strip().upper()] = _unquote(value.strip())
    return values


def load_settings(*, env_file: str | os.PathLike[str] | None = None) -> Settings:
    """Read :class:`Settings` from the process environment, explicitly.

    Each field comes from its upper-cased name (``ollama_model`` from
    ``OLLAMA_MODEL``), case-insensitively. A ``.env`` file is read only when
    given: the working directory is never assumed a safe place for secrets.

    Args:
        env_file: Optional ``.env`` file (syntax: :func:`_read_env_file`);
            environment variables win over it; a missing file is ignored.

    Raises:
        BackendConfigurationError: If the ``.env`` file cannot be read, or a
            variable holds an invalid value. The message names the variable
            but never echoes the value, which could be a secret.
    """
    found = {} if env_file is None else _read_env_file(env_file)
    found.update((name.upper(), value) for name, value in os.environ.items())
    values = {field: found[name] for name, field in _ENV_VARIABLES.items() if name in found}
    try:
        return Settings.model_validate(values)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']).upper()}: {error['msg']}"
            for error in exc.errors()
        )
        raise BackendConfigurationError(f"Invalid csv_inspector settings: {problems}") from None


def resolve_settings(settings: Settings | None) -> Settings:
    """``settings`` if given (the environment is then never read), else :func:`load_settings`.

    Raises:
        BackendConfigurationError: If an environment variable holds an invalid value.
    """
    return settings if settings is not None else load_settings()


def ensure_backend_ready(backend: LLMBackend, settings: Settings | None = None) -> None:
    """Fail fast if ``backend`` cannot be used as configured; no network or model call.

    Every inspection with a built-in invoker runs it before reading the
    source; hosts can call it at startup or in a readiness probe. The local
    backend always passes (Ollama is only known reachable when called); the
    cloud one needs credentials and the ``google-genai`` package.

    Args:
        backend: The backend to check.
        settings: The settings inspections will use. ``None`` reads them
            from the environment, as :func:`inspect_csv` does.

    Raises:
        BackendConfigurationError: If the ``[cloud]`` extra is missing or a
            setting is invalid.
        CredentialsNotConfiguredError: If the cloud backend has no usable
            credentials.
    """
    if backend is not LLMBackend.API:
        return
    resolve_settings(settings).cloud_credentials()
    try:
        sdk_installed = importlib.util.find_spec("google.genai") is not None
    except ModuleNotFoundError:
        sdk_installed = False
    if not sdk_installed:
        raise BackendConfigurationError(
            f"The 'google-genai' package is required for the 'api' backend. {CLOUD_EXTRA_HINT}"
        )
