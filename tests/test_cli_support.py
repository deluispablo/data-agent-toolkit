"""Unit tests for the csv_inspector command-line helpers."""

from __future__ import annotations

import argparse

import pytest

from cli_support import add_log_level_argument, non_negative_int, positive_int


@pytest.mark.parametrize(("raw", "expected"), [("1", 1), ("4096", 4096)])
def test_positive_int_accepts_integers_from_one(raw: str, expected: int) -> None:
    """Integers >= 1 are parsed as-is."""
    assert positive_int(raw) == expected


@pytest.mark.parametrize("raw", ["0", "-5", "abc", "1.5"])
def test_positive_int_rejects_everything_else(raw: str) -> None:
    """Zero, negatives and non-integers are rejected with an argparse error."""
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int(raw)


@pytest.mark.parametrize(("raw", "expected"), [("0", 0), ("8192", 8192)])
def test_non_negative_int_accepts_zero_and_up(raw: str, expected: int) -> None:
    """Zero is a valid value (e.g. to disable tail sampling)."""
    assert non_negative_int(raw) == expected


@pytest.mark.parametrize("raw", ["-1", "x"])
def test_non_negative_int_rejects_negatives_and_non_integers(raw: str) -> None:
    """Negatives and non-integers are rejected with an argparse error."""
    with pytest.raises(argparse.ArgumentTypeError):
        non_negative_int(raw)


def test_add_log_level_argument_defaults_to_info_and_restricts_choices() -> None:
    """The shared ``--log-level`` option defaults to INFO and rejects unknown levels."""
    parser = argparse.ArgumentParser()
    add_log_level_argument(parser)

    assert parser.parse_args([]).log_level == "INFO"
    assert parser.parse_args(["--log-level", "DEBUG"]).log_level == "DEBUG"
    with pytest.raises(SystemExit):
        parser.parse_args(["--log-level", "TRACE"])
