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
)

# How models spell a tab or "no escape character" instead of the value itself.
_TAB_SPELLINGS = frozenset({"\\t", "tab"})
_NO_ESCAPE_SPELLINGS = frozenset({"", "null", "none"})

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
        name: The column name as it appears in the header row.
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
        header_row_index: Zero-based index of the row containing the real
            column names; equivalently, the number of preamble lines (export
            banners, comments, blank lines) to skip before the header.
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
    header_row_index: int = Field(
        ge=0,
        description="Zero-based index of the row containing the real column names.",
    )
    footer_lines: list[str] = Field(default_factory=list)
    columns: list[ColumnSchema]
    confidence: float = Field(ge=0.0, le=1.0)
    notes: str | None = None

    @field_validator("delimiter", "quotechar", "escapechar", mode="before")
    @classmethod
    def _normalize_dialect_character(cls, value: object, info: ValidationInfo) -> object:
        """Map common spellings of dialect characters to the character itself.

        Small models often write a tab as ``"tab"`` or as a backslash followed
        by ``t``, and "no escape character" as ``""`` or ``"null"``.
        Those are mapped to a real tab and to ``None``; anything else that is
        not exactly one character is rejected, so a malformed answer fails
        validation (and the fallback model runs) instead of breaking
        ``csv``/pandas downstream.
        """
        if not isinstance(value, str):
            return value
        if value.lower() in _TAB_SPELLINGS:
            return "\t"
        if info.field_name == "escapechar" and value.strip().lower() in _NO_ESCAPE_SPELLINGS:
            return None
        if len(value) != 1:
            raise ValueError(f"must be exactly one character, got {value!r}")
        return value

    @computed_field  # type: ignore[prop-decorator]
    @property
    def footer_rows_to_skip(self) -> int:
        """Number of trailing footer rows to discard, i.e. ``len(footer_lines)``."""
        return len(self.footer_lines)
