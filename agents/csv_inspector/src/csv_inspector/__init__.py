r"""csv_inspector: LLM-assisted inspection of large, messy CSV/TSV sources.

Infers the encoding, dialect, header row, footer lines and a preliminary
column schema of a delimited source from small, bounded head/tail samples,
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
from ._config import Settings, load_settings
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
from ._models import ColumnSchema, ColumnType, CSVInspectionResult
from ._sampling import CSVSource

try:
    __version__ = version("csv-inspector")
except PackageNotFoundError:  # pragma: no cover - running from a source tree, not installed.
    __version__ = "0.0.0+unknown"

# Library logging etiquette: no handlers, no basicConfig; the host decides.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "BackendConfigurationError",
    "CSVInspectionResult",
    "CSVInspectorError",
    "CSVSource",
    "ColumnSchema",
    "ColumnType",
    "CredentialsNotConfiguredError",
    "EmptySampleError",
    "FileSampleReadError",
    "InspectionFailedError",
    "InspectionTimeoutError",
    "LLMBackend",
    "ModelInvocationError",
    "ModelTimeoutError",
    "ResponseParsingError",
    "SchemaValidationError",
    "Settings",
    "__version__",
    "ainspect_csv",
    "inspect_csv",
    "load_settings",
]
