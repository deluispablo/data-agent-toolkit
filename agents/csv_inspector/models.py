"""Pydantic data models for the csv_inspector agent.

These models define the strict, structured contract returned by the agent so
downstream pipelines (PySpark, BigQuery, Dataform, etc.) can consume the
inspection result without ad-hoc parsing.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator


class ColumnSchema(BaseModel):
    """Preliminary schema inferred for a single CSV column.

    Attributes:
        name: The column name as it appears in the header row.
        inferred_type: The inferred logical type (e.g. ``string``,
            ``integer``, ``float``, ``date``, ``boolean``).
        nullable: Whether the column is expected to contain missing values.
        example_values: A small sample of raw values observed for this column.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    inferred_type: str = Field(
        description="Inferred logical type: string, integer, float, date, boolean, etc."
    )
    nullable: bool = True
    example_values: list[str] = Field(default_factory=list)

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

    @computed_field  # type: ignore[prop-decorator]
    @property
    def footer_rows_to_skip(self) -> int:
        """Number of trailing footer rows to discard, i.e. ``len(footer_lines)``."""
        return len(self.footer_lines)
