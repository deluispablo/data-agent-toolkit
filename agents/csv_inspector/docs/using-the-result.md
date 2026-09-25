# Using the result to read the file

A `CSVInspectionResult` describes the file; it does not read it. This page
shows how to turn it into reader options for the stdlib `csv` module,
pandas and PySpark. The mapping has traps that return wrong data **without
any error**, so read the rules below before writing your own.

## How the result is grounded

Small local models reliably *recognize* headers and footers
but count and copy lines poorly: they miscount preamble lines, paraphrase
column names ("Importe" as "Monto"), and drop blank lines or skip a footer
line. The model's answer is therefore used as a key to recompute positions
deterministically from the sampled text:

- **Delimiter:** kept when it splits at least two head lines into the same
  number (2 or more) of fields. Otherwise (e.g. `,` for a tab-separated
  file, even one whose values hold a comma) it is replaced by the usual
  delimiter (`,`, `;`, tab, `|`) that splits the most lines that way, when
  exactly one does. Ties and one-column files keep the model's answer.
- **Header:** the head line whose fields equal the inferred column names;
  failing that, the first line with as many fields as inferred columns,
  followed by a line of the same shape, that shares at least one name with
  the model's answer. Its index becomes `header_row_index`, and its fields
  replace any paraphrased column names. Skipped when the model reports
  `has_header=false`: a header-less file keeps its positional names
  (`column_1`, ...) and `header_row_index=None`. When no header line
  anchors and the model's header at row 0 is shaped like the row below it
  (the same integer, decimal, date, empty or text shape per field, with at
  least half the fields non-text unless the shapes are identical, and no
  field equal to a model's column name), that row is data: the result is
  corrected to `has_header=false` with positional names. The test reads
  only the sample, never the model's `example_values`.
- **Footer:** the earliest reported footer line (by last occurrence) that
  really appears at the end of the source, taken verbatim through to the
  end, and extended backwards over blank separators and totals-labelled rows
  (`TOTAL`, `Subtotal`, `Total registros: 250`, `Suma`...).

A header that cannot be anchored is returned as the model reported it. A
reported footer that does not occur at the sampled end of the source is
dropped, since keeping it would make readers skip real data rows. Grounding
never promotes an unlabelled data row to a footer.

## What is not serialized

`result.usage` (the model, tokens, latency and attempts of the inspection;
see the README's Usage section) describes the call, not the file. It is
never part of `model_dump()`, `model_dump_json()` or the JSON Schema, so
the JSON a pipeline stores or forwards is the same with or without it. Read
it from the Python object right after the call if you want to keep it.

## Rules that apply to every reader

- **`has_header=False` means the first row is data.** `header_row_index`
  is then `None` and the column names are positional (`column_1`, ...):
  read with "no header" and name the columns yourself. A header-less file
  with preamble lines above its data is not described (known limitation).
- **`header_row_index` is a physical line count.** It is the number of
  lines before the header, blank lines included. Skip that many *lines*
  before parsing. Do not pass it as a "header row" option: pandas'
  `header=` and similar options count rows after dropping blank and
  comment lines, so they land one row too low for each blank preamble line.
- **`footer_rows_to_skip` is also a physical line count**, equal to
  `len(footer_lines)`. Blank separator lines before a totals row or an
  "end of report" marker are part of it.
- Both counts assume one physical line per record in the preamble and the
  footer. Data rows in between may hold quoted fields with embedded
  newlines; the recipes below handle that except where noted.
- **`encoding` is a Python codec name** (`utf-8`, `UTF-8-SIG`,
  `Windows-1252`, `ISO-8859-1`, ...). The stdlib and pandas accept it as
  is. Other engines may spell it differently; see [Encoding names](#encoding-names).

## stdlib `csv`

Skip the preamble lines on the open file, then let `csv.reader` parse the
rest. The last `footer_rows_to_skip` rows are held back in a small buffer,
so the file is streamed and never loaded in full. This recipe is run by the
test suite.

```python
import csv
from collections import deque
from collections.abc import Iterator
from pathlib import Path

from csv_inspector import CSVInspectionResult


def read_rows(path: Path, result: CSVInspectionResult) -> Iterator[list[str]]:
    """Yield the header row (if any), then every data row, skipping preamble and footer."""
    with path.open(encoding=result.encoding, newline="") as file:
        for _ in range(result.header_row_index or 0):  # None: header-less file
            file.readline()
        reader = csv.reader(
            file,
            delimiter=result.delimiter,
            quotechar=result.quotechar,
            escapechar=result.escapechar,
            doublequote=result.doublequote,
        )
        pending: deque[list[str]] = deque()
        for row in reader:
            pending.append(row)
            if len(pending) > result.footer_rows_to_skip:
                row = pending.popleft()
                if row:  # blank lines between data rows parse as []
                    yield row
```

The footer rows are dropped *before* blank rows are filtered out, so a
blank separator line in the footer is counted correctly.

## pandas

```python
import pandas as pd

df = pd.read_csv(
    path,
    encoding=result.encoding,
    sep=result.delimiter,
    quotechar=result.quotechar,
    escapechar=result.escapechar,
    doublequote=result.doublequote,
    skiprows=result.header_row_index or 0,  # physical lines: NOT header=
    header=0 if result.has_header else None,
    names=None if result.has_header else [column.name for column in result.columns],
    skipfooter=result.footer_rows_to_skip,
    engine="python" if result.footer_rows_to_skip else "c",
)
```

- A header-less file (`has_header=False`) is read with `header=None`
  and the positional names from `result.columns`.
- Use `skiprows=result.header_row_index` with `header=0`. An integer
  `skiprows` counts physical lines, blank ones included, which matches
  `header_row_index`. `header=result.header_row_index` does not: with the
  default `skip_blank_lines=True` it skips blank lines first and picks the
  wrong row as the header.
- `skipfooter` also counts physical lines, including blank footer lines,
  but only the `python` engine supports it. That engine is slower, so the
  snippet only switches to it when there is a footer to drop.
- Column types are left to pandas. `result.columns` gives the names and a
  preliminary type if you want to pass `dtype=` or `parse_dates=`.

## PySpark

Spark's CSV reader maps the dialect directly, but it cannot skip preamble
or footer lines.

| Result field | `spark.read.csv` option |
| --- | --- |
| `delimiter` | `sep` |
| `quotechar` | `quote` |
| `doublequote=True` | `escape` set to the quote character (Spark's default escape is `\`) |
| `escapechar` | `escape` |
| `encoding` | `encoding`, as a Java charset name (see below) |

```python
if result.escapechar is not None:
    escape = result.escapechar
elif result.doublequote:
    escape = result.quotechar  # RFC 4180: "" inside a quoted field
else:
    escape = "\u0000"  # no escaping at all; Spark would default to "\"

options = {
    "sep": result.delimiter,
    "quote": result.quotechar,
    "escape": escape,
    "header": result.has_header,  # False: Spark names the columns _c0, _c1, ...
}
```

**No preamble or footer** (`header_row_index` is `0` or `None` and
`footer_rows_to_skip == 0`): read the file directly.

```python
spark_encoding = "UTF-8"  # from result.encoding, see Encoding names
df = (
    spark.read.options(**options)
    .option("encoding", spark_encoding)
    .option("multiLine", True)  # quoted fields may span lines
    .csv(path)
)
```

**Preamble or footer, UTF-8 file:** drop the lines by position, then parse
the remaining lines as CSV. Spark's line-based text reader only decodes
UTF-8, and each record must fit on one physical line (there is no
`multiLine` for this path).

```python
lines = spark.sparkContext.textFile(path)
total = lines.count()
first, end = result.header_row_index or 0, total - result.footer_rows_to_skip
kept = lines.zipWithIndex().filter(lambda pair: first <= pair[1] < end).keys()
df = spark.read.options(**options).csv(kept)
```

**Otherwise** (another encoding, or multi-line records together with a
preamble or footer): rewrite the file first as clean UTF-8 with the
[stdlib recipe](#stdlib-csv) and `csv.writer`, then read it with the
first snippet.

## Encoding names

`result.encoding` is a Python codec name. Python and pandas accept it
directly. Spark (Java charsets) and BigQuery (`--encoding` / `encoding`)
need their own spelling:

| Python (canonical name) | Spark / Java | BigQuery |
| --- | --- | --- |
| `utf-8` | `UTF-8` | `UTF-8` |
| `utf-8-sig` (UTF-8 with a BOM) | `UTF-8` (strip `﻿` from the first column name if it appears) | `UTF-8` |
| `iso8859-1` (latin-1) | `ISO-8859-1` | `ISO-8859-1` |
| `cp1252` (windows-1252) | `windows-1252` | not supported: transcode to UTF-8 first |
| `utf-16`, `utf-16-le`, `utf-16-be` | `UTF-16`, `UTF-16LE`, `UTF-16BE` | `UTF-16LE`, `UTF-16BE` |

Normalize the name before looking it up, because detectors and models vary
the spelling: `codecs.lookup(result.encoding).name` returns Python's
canonical name, as in the first column (`Windows-1252` becomes `cp1252`,
`latin-1` becomes `iso8859-1`).
