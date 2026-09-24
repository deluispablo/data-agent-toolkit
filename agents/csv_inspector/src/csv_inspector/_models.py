"""Pydantic data models for the csv_inspector agent.

These models define the strict, structured contract returned by the agent so
downstream pipelines (PySpark, BigQuery, Dataform, etc.) can consume the
inspection result without ad-hoc parsing.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic.json_schema import SkipJsonSchema

# How models spell a tab or "no escape character" instead of the value itself.
_TAB_SPELLINGS = frozenset({"\\t", "tab"})
_NO_ESCAPE_SPELLINGS = frozenset({"", "null", "none"})

# Characters that end a row: csv and pandas reject them as dialect characters.
_LINE_BREAKS = frozenset({"\r", "\n"})

ColumnType = Literal["string", "integer", "float", "date", "datetime", "boolean"]
"""The closed vocabulary of ``ColumnSchema.inferred_type`` values."""

# How models spell a type outside the vocabulary. Lookup is case-insensitive.
_COLUMN_TYPE_ALIASES: Mapping[str, ColumnType] = MappingProxyType(
    {
        "int": "integer",
        "bigint": "integer",
        "int64": "integer",
        "number": "float",
        "decimal": "float",
        "double": "float",
        "numeric": "float",
        "text": "string",
        "str": "string",
        "varchar": "string",
        "bool": "boolean",
        "timestamp": "datetime",
    }
)


class ColumnSchema(BaseModel):
    """Preliminary schema inferred for a single CSV column.

    Attributes:
        name: The column name as it appears in the header row, or a
            positional name (``column_1``, ``column_2``, ...) when the
            file has no header row.
        inferred_type: The inferred logical type, one of ``string``,
            ``integer``, ``float``, ``date``, ``datetime`` or ``boolean``.
        nullable: Whether the column is expected to contain missing values.
        example_values: A small sample of raw values observed for this column.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    inferred_type: ColumnType = Field(description="Inferred logical type of the column.")
    nullable: bool = True
    example_values: list[str] = Field(default_factory=list)

    @field_validator("inferred_type", mode="before")
    @classmethod
    def _normalize_inferred_type(cls, value: object) -> object:
        """Map the model's type word onto the closed vocabulary.

        Small models answer ``int``, ``number``, ``text`` and the like instead
        of the requested names. Common aliases are mapped, case-insensitively,
        and any other string becomes ``string``, the safe type, rather than
        failing the whole inspection over one column.
        """
        if not isinstance(value, str):
            return value
        normalized = value.strip().lower()
        if normalized in get_args(ColumnType):
            return normalized
        return _COLUMN_TYPE_ALIASES.get(normalized, "string")

    @field_validator("example_values", mode="before")
    @classmethod
    def _stringify_scalar_examples(cls, value: object) -> object:
        """Accept JSON scalars as raw example values.

        Models often emit numeric examples as JSON numbers (``1447.44``)
        rather than strings. Examples are raw text by contract, so scalars
        are rendered as their JSON text (``null`` as ``""``) instead of
        failing the whole inspection over a cosmetic field.
        """
        if not isinstance(value, list):
            return value
        return [
            "" if item is None else json.dumps(item) if isinstance(item, int | float) else item
            for item in value
        ]


class Usage(BaseModel):
    """What the model phase of one successful inspection cost.

    Attached to the result as ``CSVInspectionResult.usage`` and never
    serialized: it describes the call, not the file.

    Attributes:
        model: The model whose answer was kept.
        prompt_tokens: Prompt tokens summed over every model attempt of the
            inspection (a failed primary still costs tokens), or ``None``
            when no attempt reported them (e.g. a custom model invoker).
        completion_tokens: Completion tokens, summed like ``prompt_tokens``.
        latency_seconds: Wall time of the model phase, all attempts included.
        attempts: How many models were called.
        retries: Transient cloud errors (429/503) retried within an attempt.
        load_seconds: Time Ollama spent loading the model, summed over the
            attempts, or ``None`` when no attempt reported it (cloud backend,
            custom invoker).
        prompt_version: The version of the prompt the models were sent
            (``PROMPT_VERSION``), so measurements of different prompts are
            never mixed.
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


class CSVInspectionResult(BaseModel):
    """Structured, validated result of inspecting a CSV/TSV file fragment.

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
        columns: The preliminary schema inferred for each column.
        confidence: The model's self-reported confidence, in ``[0.0, 1.0]``.
        notes: Optional free-text observations relevant to downstream parsing.
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
    footer_lines: list[str] = Field(default_factory=list)
    columns: list[ColumnSchema]
    confidence: float = Field(ge=0.0, le=1.0)
    notes: str | None = None
    # Tokens, latency and attempts of the inspection that returned this result
    # (see Usage); None on a result built any other way. Kept out of the JSON
    # Schema sent to the model and out of every dump, so the serialized result
    # keeps its contract. Documented here, not in the docstring above: that
    # docstring is the schema's description, part of every cloud request.
    usage: SkipJsonSchema[Usage | None] = Field(default=None, exclude=True)

    @field_validator("delimiter", "quotechar", "escapechar", mode="before")
    @classmethod
    def _normalize_dialect_character(cls, value: object, info: ValidationInfo) -> object:
        """Map common spellings of dialect characters to the character itself.

        Small models often write a tab as ``"tab"`` or as a backslash followed
        by ``t``, and "no escape character" as ``""`` or ``"null"``.
        Those are mapped to a real tab and to ``None``. For unquoted files the
        same spellings (and JSON ``null``) come back as the quote character;
        they map to the default ``'"'``, which is inert for ``csv``, pandas,
        PySpark and BigQuery when it never occurs in the file. Anything else
        that is not exactly one character is rejected, so a malformed answer
        fails validation (and the fallback model runs) instead of breaking
        ``csv``/pandas downstream.
        """
        if value is None and info.field_name == "quotechar":
            return '"'
        if not isinstance(value, str):
            return value
        if value.lower() in _TAB_SPELLINGS:
            return "\t"
        if info.field_name == "escapechar" and value.strip().lower() in _NO_ESCAPE_SPELLINGS:
            return None
        # Strip spaces only: a line break stays a (rejected) quote character.
        if info.field_name == "quotechar" and value.strip(" ").lower() in _NO_ESCAPE_SPELLINGS:
            return '"'
        if len(value) != 1:
            raise ValueError(f"must be exactly one character, got {value!r}")
        return value

    @model_validator(mode="before")
    @classmethod
    def _escaped_quote_means_doublequote(cls, data: object) -> object:
        """Read an escape character equal to the quote character as doubled quotes.

        Models describe RFC 4180 quoting (``""`` inside a quoted field) as
        "the quote is escaped by a quote" and answer ``escapechar='"'``.
        ``csv`` rejects that dialect, but the meaning is clear, so it is
        mapped to ``escapechar=None, doublequote=True``.
        """
        if not isinstance(data, dict):
            return data
        escapechar = data.get("escapechar")
        if escapechar is not None and escapechar == data.get("quotechar", '"'):
            return {**data, "escapechar": None, "doublequote": True}
        return data

    @model_validator(mode="before")
    @classmethod
    def _infer_has_header(cls, data: object) -> object:
        """Read a null (or ``-1``) header row index as "no header row".

        Models answer ``null`` or ``-1`` for a header-less file, often
        without ``has_header``. ``-1`` is taken as ``null``, and a missing
        ``has_header`` follows the header row index. An explicit
        ``has_header`` is kept, so a contradiction still fails validation.
        """
        if not isinstance(data, dict) or "header_row_index" not in data:
            return data
        index = data["header_row_index"]
        if index == -1 and not isinstance(index, bool):
            data = {**data, "header_row_index": None}
        if "has_header" not in data:
            data = {**data, "has_header": data["header_row_index"] is not None}
        return data

    @model_validator(mode="after")
    def _check_header(self) -> CSVInspectionResult:
        """Require a header row index exactly when the file has a header."""
        if self.has_header and self.header_row_index is None:
            raise ValueError("header_row_index is required when has_header is true")
        if not self.has_header and self.header_row_index is not None:
            raise ValueError("header_row_index must be null when has_header is false")
        return self

    @model_validator(mode="after")
    def _check_dialect(self) -> CSVInspectionResult:
        """Reject a dialect that ``csv`` and pandas cannot read.

        Each character may be valid alone, but a line break cannot separate
        or quote fields, and the delimiter, quote and escape characters must
        all differ. Failing validation moves on to the fallback model instead
        of breaking grounding and every reader downstream.
        """
        characters = {
            "delimiter": self.delimiter,
            "quotechar": self.quotechar,
            "escapechar": self.escapechar,
        }
        for name, character in characters.items():
            if character in _LINE_BREAKS:
                raise ValueError(f"{name} cannot be a line break, got {character!r}")
        if self.delimiter in (self.quotechar, self.escapechar):
            raise ValueError(
                f"delimiter {self.delimiter!r} must differ from quotechar and escapechar"
            )
        if self.escapechar == self.quotechar:
            raise ValueError(f"escapechar and quotechar must differ, got {self.quotechar!r}")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def footer_rows_to_skip(self) -> int:
        """Number of trailing footer rows to discard, i.e. ``len(footer_lines)``."""
        return len(self.footer_lines)
