"""Pydantic data models for the csv_inspector agent.

These models define the strict, structured contract returned by the agent so
downstream pipelines (PySpark, BigQuery, Dataform, etc.) can consume the
inspection result without ad-hoc parsing.
"""

from __future__ import annotations

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

# A confidence above 1 and up to this is a percentage (see _ModelAnswer).
_PERCENT = 100


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


class _StrictDialect(BaseModel):
    """The strict checks the model's answer and the result share.

    Each dialect character is exactly one character, none is a line break,
    the delimiter, quote and escape characters all differ (``csv`` and
    pandas cannot read anything else), and ``header_row_index`` is set
    exactly when ``has_header`` is true. Failing validation moves on to the
    fallback model instead of breaking grounding and every reader
    downstream. The fields common to both come first, in the order both
    serialize them.
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

    Private: hosts receive :class:`CSVInspectionResult`, which
    ``ground_in_samples`` builds from this answer and the samples. Its JSON
    Schema, stripped of annotations, is the schema both backends send
    (``_prompt.response_schema``). The fields are those of the result,
    except that the footer is asked for as one anchor line,
    ``footer_first_line`` (the first non-blank line after the last data
    row, or ``None``): grounding re-reads the whole footer from the file.

    Validation is lenient where small models are predictably sloppy (a tab
    spelled ``"tab"``, a ``-1`` header index, an escaped quote meaning
    doubled quotes, padded column names) and strict where a reader would
    break (see :class:`_StrictDialect`). An answer in the pre-0.4
    shape, with ``footer_lines`` instead of ``footer_first_line`` (a custom
    ``model_invoker`` written for an older release), is still accepted: its
    first non-blank footer line is the anchor.
    """

    footer_first_line: str | None = None
    columns: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("delimiter", "quotechar", "escapechar", mode="before")
    @classmethod
    def _normalize_dialect_character(cls, value: object, info: ValidationInfo) -> object:
        """Map common spellings of dialect characters to the character itself.

        Small models often write a tab as ``"tab"`` or as a backslash followed
        by ``t``, and "no escape character" as ``""`` or ``"null"``.
        Those are mapped to a real tab and to ``None``. A line break given
        as the delimiter (a one-column file read as "one field per line")
        maps to ``,``, which splits nothing there; grounding still replaces
        it when another delimiter splits the head. For unquoted files the
        same spellings (and JSON ``null``) come back as the quote character;
        they map to the default ``'"'``, which is inert for ``csv``, pandas,
        PySpark and BigQuery when it never occurs in the file. Anything else
        is left to the strict check, which rejects a value that is not
        exactly one character, so a malformed answer fails validation (and
        the fallback model runs) instead of breaking ``csv``/pandas
        downstream.
        """
        if value is None and info.field_name == "quotechar":
            return '"'
        if not isinstance(value, str):
            return value
        if value.lower() in _TAB_SPELLINGS:
            return "\t"
        if info.field_name == "delimiter" and value and not value.strip("\r\n"):
            value = ","
        if info.field_name == "escapechar" and value.strip().lower() in _NO_ESCAPE_SPELLINGS:
            return None
        # Strip spaces only: a line break stays a (rejected) quote character.
        if info.field_name == "quotechar" and value.strip(" ").lower() in _NO_ESCAPE_SPELLINGS:
            return '"'
        return value

    @field_validator("columns", mode="before")
    @classmethod
    def _strip_column_names(cls, value: object) -> object:
        """Strip surrounding whitespace from each column name.

        Empty names (a pandas index column is written as ``""``) and
        duplicate names are kept: they are what the file holds. Anything
        that is not a list of strings is left to fail validation.
        """
        if not isinstance(value, list):
            return value
        return [name.strip() if isinstance(name, str) else name for name in value]

    @model_validator(mode="before")
    @classmethod
    def _footer_lines_as_first_line(cls, data: object) -> object:
        """Read a pre-0.4 ``footer_lines`` list as its first footer line.

        The first non-blank line is the anchor; a list of blank lines only
        is a blank anchor, and an empty list is ``None``. An explicit
        ``footer_first_line`` wins.
        """
        if not isinstance(data, dict) or "footer_lines" not in data:
            return data
        lines = data["footer_lines"]
        data = {key: value for key, value in data.items() if key != "footer_lines"}
        if "footer_first_line" in data or not isinstance(lines, list):
            return data
        texts = [line for line in lines if isinstance(line, str)]
        anchor = next((line for line in texts if line.strip()), "" if texts else None)
        return {**data, "footer_first_line": anchor}

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
        ``has_header`` follows the header row index. ``has_header: true``
        with a null index is ambiguous (models answer it for real headers
        and for header-less files alike): it is read as row 0, and grounding
        decides, anchoring the names or finding row 0 shaped like data.
        ``has_header: false`` with an index still fails validation.
        """
        if not isinstance(data, dict) or "header_row_index" not in data:
            return data
        index = data["header_row_index"]
        if index == -1 and not isinstance(index, bool):
            data = {**data, "header_row_index": None}
        if "has_header" not in data:
            data = {**data, "has_header": data["header_row_index"] is not None}
        elif data["has_header"] is True and data["header_row_index"] is None:
            data = {**data, "header_row_index": 0}
        return data

    @field_validator("confidence", mode="before")
    @classmethod
    def _percent_confidence(cls, value: object) -> object:
        """Read a confidence above 1 and up to 100 as a percentage.

        A response schema cannot make a model respect ``maximum``; small
        models answer ``90`` or ``100``.
        """
        if isinstance(value, int | float) and not isinstance(value, bool) and 1 < value <= _PERCENT:
            return value / _PERCENT
        return value

    @field_validator("footer_first_line", mode="before")
    @classmethod
    def _first_line_only(cls, value: object) -> object:
        """Keep the first non-blank line of an anchor that spans several lines.

        Asked for one line, models sometimes copy the whole footer. The first
        non-blank line is the anchor; blank lines only are a blank anchor.
        """
        if not isinstance(value, str) or ("\n" not in value and "\r" not in value):
            return value
        return next((line for line in value.splitlines() if line.strip()), "")


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
    # Tokens, latency and attempts of the inspection that returned this result
    # (see Usage); None on a result built any other way. Kept out of the JSON
    # Schema and out of every dump, so the serialized result keeps its
    # contract. Documented here, not in the docstring above: that docstring
    # is the schema's description.
    usage: SkipJsonSchema[Usage | None] = Field(default=None, exclude=True)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def footer_rows_to_skip(self) -> int:
        """Number of trailing footer rows to discard, i.e. ``len(footer_lines)``."""
        return len(self.footer_lines)
