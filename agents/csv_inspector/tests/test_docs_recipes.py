"""Run the stdlib reader recipe from ``docs/using-the-result.md``.

The recipe is executed straight from the Markdown, so the documented code
cannot drift from what these tests check.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import cast

import pytest

from csv_inspector import CSVInspectionResult
from generate_samples import CASES, SampleCase

DOC_PATH = Path(__file__).resolve().parent.parent / "docs" / "using-the-result.md"

ReadRows = Callable[[Path, CSVInspectionResult], Iterator[list[str]]]


def _load_read_rows() -> ReadRows:
    """Execute the doc's ``python`` block that defines ``read_rows``."""
    blocks = re.findall(r"```python\n(.*?)```", DOC_PATH.read_text(encoding="utf-8"), re.DOTALL)
    (recipe,) = [block for block in blocks if "def read_rows(" in block]
    namespace: dict[str, object] = {}
    exec(recipe, namespace)  # trusted, repo-owned documentation
    return cast("ReadRows", namespace["read_rows"])


read_rows = _load_read_rows()


def _result(
    encoding: str, delimiter: str, header_row_index: int | None, footer_lines: list[str]
) -> CSVInspectionResult:
    return CSVInspectionResult(
        encoding=encoding,
        delimiter=delimiter,
        has_header=header_row_index is not None,
        header_row_index=header_row_index,
        footer_lines=footer_lines,
        columns=["unused"],  # read_rows reads the dialect and positions only
        confidence=1.0,
    )


_PREAMBLE_OR_FOOTER_CASES = [
    case
    for case in CASES
    if "footer_lines" in case.expected
    and (case.expected["header_row_index"] or case.expected["footer_lines"])
]


def test_fixture_selection_covers_preamble_and_footer() -> None:
    """The parametrized fixtures include a preamble and a blank footer line."""
    assert any(case.expected["header_row_index"] for case in _PREAMBLE_OR_FOOTER_CASES)
    assert any("" in case.expected["footer_lines"] for case in _PREAMBLE_OR_FOOTER_CASES)


@pytest.mark.parametrize("case", _PREAMBLE_OR_FOOTER_CASES, ids=lambda case: case.filename)
def test_read_rows_skips_preamble_and_footer_of_fixture(case: SampleCase, tmp_path: Path) -> None:
    """On a fixture's ground truth, the recipe yields exactly header + data rows."""
    expected = case.expected
    # The first alternative of an "A or B" encoding label is a codec name.
    codec = expected["encoding"].split(" or ")[0].strip()
    path = tmp_path / case.filename
    path.write_bytes(case.raw_bytes)
    result = _result(
        codec,
        expected["delimiter"],
        expected["header_row_index"],
        expected["footer_lines"],
    )

    rows = list(read_rows(path, result))

    lines = case.raw_bytes.decode(codec).splitlines()
    body = lines[result.header_row_index or 0 : len(lines) - result.footer_rows_to_skip]
    oracle = [row for row in csv.reader(body, delimiter=result.delimiter) if row]
    assert rows == oracle
    assert {len(row) for row in rows} == {len(rows[0])}


def test_read_rows_counts_blank_lines_and_keeps_multiline_fields(tmp_path: Path) -> None:
    """Blank preamble/footer lines count; quoted newlines and blank data lines are handled."""
    path = tmp_path / "report.csv"
    path.write_bytes(b'# export\n\nA,B\n1,"x\ny"\n\n2,z\n\nTOTAL,3\n')
    result = _result("utf-8", ",", 2, ["", "TOTAL,3"])

    assert list(read_rows(path, result)) == [["A", "B"], ["1", "x\ny"], ["2", "z"]]


def test_read_rows_without_preamble_or_footer(tmp_path: Path) -> None:
    """With nothing to skip, every row is returned."""
    path = tmp_path / "plain.csv"
    path.write_bytes(b"A;B\n1;2\n")

    assert list(read_rows(path, _result("utf-8", ";", 0, []))) == [["A", "B"], ["1", "2"]]


def test_read_rows_of_a_header_less_file(tmp_path: Path) -> None:
    """has_header=False: every row is data, nothing is skipped (issue #94)."""
    path = tmp_path / "data_only.csv"
    path.write_bytes(b"2024-01-15,Acme,1\n2024-01-16,Beta,2\n")

    rows = list(read_rows(path, _result("utf-8", ",", None, [])))

    assert rows == [["2024-01-15", "Acme", "1"], ["2024-01-16", "Beta", "2"]]
