"""Domain-specific exception hierarchy for the csv_inspector agent.

Centralizing exceptions here keeps failure modes explicit and lets callers
(CLI scripts, orchestration pipelines, other agents) catch precisely the
failure they care about instead of a bare ``Exception``.
"""

from __future__ import annotations


class CSVInspectorError(Exception):
    """Base class for all errors raised by the csv_inspector agent."""


class FileSampleReadError(CSVInspectorError):
    """Raised when the initial byte sample cannot be read from the source file."""


class EmptySampleError(CSVInspectorError):
    """Raised when the source file is empty, so there is nothing to inspect.

    Detected before any model is invoked, so an empty file never costs an
    LLM call.
    """


class ModelInvocationError(CSVInspectorError):
    """Raised when the configured LLM backend fails to return a response."""


class ResponseParsingError(CSVInspectorError):
    """Raised when the raw LLM response cannot be parsed as valid JSON."""


class SchemaValidationError(CSVInspectorError):
    """Raised when the parsed JSON payload does not satisfy the expected schema."""


class InspectionFailedError(CSVInspectorError):
    """Raised when every configured model fails to produce a valid inspection result.

    Attributes:
        attempts: Mapping of model name to the exception raised for that
            model, preserved for diagnostics and structured logging.
    """

    def __init__(self, message: str, attempts: dict[str, Exception]) -> None:
        """Initialize the error with the aggregated per-model failures.

        Args:
            message: Human-readable summary of the aggregate failure.
            attempts: Mapping of model name to the exception it raised.
        """
        super().__init__(message)
        self.attempts = attempts
