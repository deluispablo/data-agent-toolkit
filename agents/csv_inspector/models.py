"""Pydantic data models for the csv_inspector agent.

These models define the strict, structured contract returned by the agent so
downstream pipelines (PySpark, BigQuery, Dataform, etc.) can consume the
inspection result without ad-hoc parsing.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


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
            column names.
        metadata_lines: Raw lines preceding the header row (export banners,
            comments, etc.) that are not part of the tabular data.
        footer_rows_to_skip: Number of trailing rows to discard as
            non-data footers.
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
    metadata_lines: list[str] = Field(default_factory=list)
    footer_rows_to_skip: int = Field(default=0, ge=0)
    columns: list[ColumnSchema]
    confidence: float = Field(ge=0.0, le=1.0)
    notes: str | None = None
