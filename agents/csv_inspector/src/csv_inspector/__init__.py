r"""csv_inspector: LLM-assisted inspection of large, messy CSV/TSV sources.

Infers the encoding, dialect, header row, footer lines and column names
of a delimited source from small, bounded head/tail samples,
using a local Ollama model by default or Google Gemini as an opt-in.

The names exported here (see ``__all__``) are the **public, stable API**.
Every other module, and every name not listed here, is internal and may
change without notice.

Example:
    >>> from csv_inspector import inspect_csv  # doctest: +SKIP
    >>> result = inspect_csv(b"a;b\n1;2\n")  # doctest: +SKIP
    >>> result.delimiter  # doctest: +SKIP
    ';'
"""

from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError, version

from ._backends import LLMBackend
from ._config import Settings, ensure_backend_ready, load_settings
from ._exceptions import (
    BackendConfigurationError,
    CredentialsNotConfiguredError,
    CSVInspectorError,
    EmptySampleError,
    FileSampleReadError,
    InspectionFailedError,
    InspectionTimeoutError,
    ModelInvocationError,
    ModelTimeoutError,
    ResponseParsingError,
    SchemaValidationError,
)
from ._inspect import ainspect_csv, inspect_csv
from ._invokers import AsyncModelInvoker, ModelInvoker
from ._models import CSVInspectionResult, Usage
from ._sampling import DEFAULT_SAMPLE_BYTES, DEFAULT_TAIL_BYTES, MAX_SAMPLE_BYTES, CSVSource

try:
    __version__ = version("csv-inspector")
except PackageNotFoundError:  # pragma: no cover - running from a source tree, not installed.
    __version__ = "0.0.0+unknown"

# Library logging etiquette: no handlers, no basicConfig; the host decides.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "DEFAULT_SAMPLE_BYTES",
    "DEFAULT_TAIL_BYTES",
    "MAX_SAMPLE_BYTES",
    "AsyncModelInvoker",
    "BackendConfigurationError",
    "CSVInspectionResult",
    "CSVInspectorError",
    "CSVSource",
    "CredentialsNotConfiguredError",
    "EmptySampleError",
    "FileSampleReadError",
    "InspectionFailedError",
    "InspectionTimeoutError",
    "LLMBackend",
    "ModelInvocationError",
    "ModelInvoker",
    "ModelTimeoutError",
    "ResponseParsingError",
    "SchemaValidationError",
    "Settings",
    "Usage",
    "__version__",
    "ainspect_csv",
    "ensure_backend_ready",
    "inspect_csv",
    "load_settings",
]
