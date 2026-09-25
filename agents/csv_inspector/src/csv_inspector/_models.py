"""The model's answer (lenient, private) and the library's result (strict, public)."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic.json_schema import SkipJsonSchema

# How models spell a tab, or no quote or escape character, instead of writing it.
_TAB_SPELLINGS = frozenset({"\\t", "tab"})
_NO_CHARACTER_SPELLINGS = frozenset({"", "null", "none"})
# What no quote or escape character means: the inert default quote, no escape.
_NO_CHARACTER: Mapping[str, str | None] = MappingProxyType({"quotechar": '"', "escapechar": None})

# Characters that end a row: csv and pandas reject them as dialect characters.
_LINE_BREAKS = frozenset({"\r", "\n"})

# A confidence above 1 and up to this is a percentage (see _ModelAnswer).
_PERCENT = 100


class Usage(BaseModel):
    """What the model phase of one successful inspection cost.

    Attached to the result as ``CSVInspectionResult.usage`` and never
    serialized: it describes the call, not the file.

    Attributes:
        model: The model whose answer was kept.
        prompt_tokens: Prompt tokens of every attempt (a failed primary costs
            too), or ``None`` when none reported them (e.g. a custom invoker).
        completion_tokens: Completion tokens, summed like ``prompt_tokens``.
        latency_seconds: Wall time of the model phase, all attempts included.
        attempts: How many models were called.
        retries: Transient cloud errors (429/503) retried within an attempt.
        load_seconds: Time Ollama spent loading the model over the attempts,
            or ``None`` when none reported it (cloud backend, custom invoker).
        prompt_version: The ``PROMPT_VERSION`` the models were sent, so
            measurements of different prompts are never mixed.
    """

    model_config = ConfigDict(frozen=True)

    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_seconds: float
    attempts: int
    retries: int = 0
    load_seconds: float | None = None
    prompt_version: str


class _StrictDialect(BaseModel):
    """The strict checks the model's answer and the result share, on their common fields.

    One character each, no line break, all different (what ``csv`` and pandas
    can read), and ``header_row_index`` set exactly when ``has_header`` is.
    A failing answer moves on to the fallback model instead of breaking readers.
    """

    model_config = ConfigDict(frozen=True)

    encoding: str
    delimiter: str
    quotechar: str = '"'
    escapechar: str | None = None
    doublequote: bool = True
    has_header: bool = Field(
        default=True,
        description="Whether the file has a row of column names; false if its first row is data.",
    )
    header_row_index: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Zero-based index of the row containing the real column names; "
            "null exactly when has_header is false."
        ),
    )

    @field_validator("delimiter", "quotechar", "escapechar")
    @classmethod
    def _check_one_character(cls, value: str | None) -> str | None:
        """Reject a dialect character that is not exactly one character."""
        if value is not None and len(value) != 1:
            raise ValueError(f"must be exactly one character, got {value!r}")
        return value

    @model_validator(mode="after")
    def _check_header_and_dialect(self) -> _StrictDialect:
        """Require a consistent header row index and a readable dialect."""
        if self.has_header and self.header_row_index is None:
            raise ValueError("header_row_index is required when has_header is true")
        if not self.has_header and self.header_row_index is not None:
            raise ValueError("header_row_index must be null when has_header is false")
        delimiter, quotechar, escapechar = self.delimiter, self.quotechar, self.escapechar
        characters = {"delimiter": delimiter, "quotechar": quotechar, "escapechar": escapechar}
        for name, character in characters.items():
            if character in _LINE_BREAKS:
                raise ValueError(f"{name} cannot be a line break, got {character!r}")
        if delimiter in (quotechar, escapechar):
            raise ValueError(f"delimiter {delimiter!r} must differ from quotechar and escapechar")
        if escapechar == quotechar:
            raise ValueError(f"escapechar and quotechar must differ, got {quotechar!r}")
        return self


class _ModelAnswer(_StrictDialect):
    """What the model is asked to answer: the key that grounding turns into a result.

    Its JSON Schema, stripped of annotations, is the one both backends send
    (``_prompt.response_schema``). The footer is one anchor line,
    ``footer_first_line`` (the first non-blank line after the data, or
    ``None``): grounding reads the whole footer from the file. Lenient with
    predictable slips, strict where a reader would break.
    """

    footer_first_line: str | None = None
    columns: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def _read_small_model_slips(cls, data: object) -> object:
        """Map the predictable slips of small models to what they mean, in one pass.

        - An escape character equal to the quote character ("the quote is
          escaped by a quote") is RFC 4180: ``escapechar=None, doublequote=True``.
        - A ``-1`` header row index is ``null``; see :func:`_read_header_index`.
        - Dialect characters spelled out: see :func:`_read_dialect_spelling`.
        - A confidence above 1 and up to 100 is a percentage (``90``).
        - A multi-line ``footer_first_line`` keeps its first non-blank line.
        - Column names lose surrounding spaces; empty and duplicates stay.

        Anything else is left to the strict checks, so a malformed answer
        fails validation and the fallback model runs.
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)
        escapechar = data.get("escapechar")
        if escapechar is not None and escapechar == data.get("quotechar", '"'):
            data.update(escapechar=None, doublequote=True)
        if "header_row_index" in data:
            _read_header_index(data)
        for field in ("delimiter", "quotechar", "escapechar"):
            if field in data:
                data[field] = _read_dialect_spelling(field, data[field])
        confidence = data.get("confidence")
        if (
            isinstance(confidence, int | float)
            and not isinstance(confidence, bool)
            and 1 < confidence <= _PERCENT
        ):
            data["confidence"] = confidence / _PERCENT
        anchor = data.get("footer_first_line")
        if isinstance(anchor, str) and ("\n" in anchor or "\r" in anchor):
            lines = [line for line in anchor.splitlines() if line.strip()]
            data["footer_first_line"] = lines[0] if lines else ""
        if isinstance(data.get("columns"), list):
            data["columns"] = [n.strip() if isinstance(n, str) else n for n in data["columns"]]
        return data


def _read_header_index(data: dict[str, object]) -> None:
    """Read a ``-1`` index as ``null``; a missing ``has_header`` follows the index.

    ``has_header: true`` with a null index is read as row 0: grounding decides.
    """
    index = data["header_row_index"]
    if index == -1 and not isinstance(index, bool):
        data["header_row_index"] = None
    if "has_header" not in data:
        data["has_header"] = data["header_row_index"] is not None
    elif data["has_header"] is True and data["header_row_index"] is None:
        data["header_row_index"] = 0


def _read_dialect_spelling(field: str, value: object) -> object:
    """Map a spelled-out dialect character to the character itself.

    ``"tab"`` or backslash-``t`` is a tab; a delimiter of line breaks (a
    one-column file) is ``,``; ``null``, ``""``, ``"null"`` or ``"none"`` is
    the inert default quote ``'"'``, or no escape character.
    """
    if value is None:
        return _NO_CHARACTER.get(field)
    if not isinstance(value, str):
        return value
    if value.lower() in _TAB_SPELLINGS:
        return "\t"
    if field == "delimiter":
        return "," if value and not value.strip("\r\n") else value
    # Strip spaces only from a quote character: a line break stays (and is rejected).
    spelling = value.strip(" " if field == "quotechar" else None).lower()
    return _NO_CHARACTER[field] if spelling in _NO_CHARACTER_SPELLINGS else value


class CSVInspectionResult(_StrictDialect):
    """Structured, validated result of inspecting a CSV/TSV file fragment.

    The library builds it from the model's answer grounded in the samples;
    the model never answers this type directly. Validation is strict: each
    dialect character is exactly one character, the delimiter, quote and
    escape characters differ and none is a line break, and
    ``header_row_index`` is set exactly when ``has_header`` is true.

    Attributes:
        encoding: The detected character encoding (e.g. ``utf-8``, ``latin-1``).
        delimiter: The field delimiter character (e.g. ``,`` or ``;``).
        quotechar: The character used to quote fields containing the delimiter.
        escapechar: The escape character, if any, used to escape the quote
            character inside a field.
        doublequote: Whether embedded quote characters are escaped by
            doubling them, per RFC 4180.
        has_header: Whether the file has a row of column names. ``False``
            for a header-less file, whose first row is already data.
        header_row_index: Zero-based index of the row containing the real
            column names; equivalently, the number of preamble lines (export
            banners, comments, blank lines) to skip before the header.
            ``None`` exactly when ``has_header`` is ``False``; a header-less
            file with preamble lines is not described (known limitation).
        footer_lines: Raw trailing lines (totals, summary rows, "end of
            report" markers, generation timestamps, blank separator lines)
            that follow the last data row, in file order.
        footer_rows_to_skip: Number of trailing rows to discard as non-data
            footers. Derived from ``footer_lines`` rather than inferred
            separately, so the two can never disagree.
        columns: The column names as written in the header row, in file
            order, or positional names (``column_1``, ``column_2``, ...) when
            ``has_header`` is ``False``. Surrounding whitespace in the model's
            answer is stripped; empty names and duplicates are kept, as they
            are in the file.
        confidence: The model's self-reported confidence, in ``[0.0, 1.0]``.
    """

    footer_lines: list[str] = Field(default_factory=list)
    columns: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    # What the inspection that returned this result cost (see Usage), else None.
    # Out of the JSON Schema and every dump, so the serialized contract holds;
    # documented here because the docstring above is the schema's description.
    usage: SkipJsonSchema[Usage | None] = Field(default=None, exclude=True)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def footer_rows_to_skip(self) -> int:
        """Number of trailing footer rows to discard, i.e. ``len(footer_lines)``."""
        return len(self.footer_lines)
