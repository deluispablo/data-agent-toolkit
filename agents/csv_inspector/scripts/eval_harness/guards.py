"""Which fixtures run, and the quota guards that keep a cloud run within a free tier.

``--subset``, ``--fixture``, ``--category`` and ``--max-fixtures`` pick the
fixtures; ``--max-calls`` (:class:`CallBudget`) and ``--rpm``
(:class:`RateLimiter`) bound the model requests, counted in the worst case
before a fixture starts.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from csv_inspector import LLMBackend

if TYPE_CHECKING:
    from csv_inspector import Usage

logger = logging.getLogger(__name__)

SAMPLES_DIR = Path(__file__).resolve().parents[2] / "samples"
MANIFEST_PATH = SAMPLES_DIR / "manifest.json"

SUBSETS: dict[str, tuple[str, ...]] = {
    # Iterate here (--repeat 2), prove on the full catalog (--repeat 3). Every
    # category, every footer kind, header-less files, a 40-column file, the
    # tab-with-commas files, the blank-name header and the 0.3.0 baseline misses.
    "quick": (
        "delimiter_semicolon_decimal_comma.csv",
        "delimiter_tab_commas_quoted_header.tsv",
        "empty_file.csv",
        "encoding_cp1252_tail_only.csv",
        "footer_end_marker.csv",
        "gen_combo_eu_legacy.csv",
        "gen_encoding_utf16le_lf.csv",
        "gen_encoding_utf8_lf.csv",
        "gen_footer_blank_totals_wide.csv",
        "gen_footer_marker_narrow.csv",
        "gen_footer_none_narrow.csv",
        "gen_footer_timestamp_narrow.csv",
        "gen_footer_totals_narrow.csv",
        "gen_headerless_marker.csv",
        "header_duplicate_and_blank_names.csv",
        "header_metadata_banner.csv",
        "header_none_data_only.csv",
        "header_years.csv",
        "numeric_european_format.csv",
        "quoting_backslash_escape.csv",
        "single_column.csv",
    ),
    # One or two fixtures per category, none a known limitation: fits a day
    # of a free-tier cloud key with --max-calls 18 (docs/evaluation.md).
    "cloud": (
        "delimiter_semicolon_decimal_comma.csv",
        "delimiter_tab_commas_quoted_header.tsv",
        "encoding_cp1252_tail_only.csv",
        "encoding_utf16le_bom.csv",
        "header_and_footer_combined.csv",
        "footer_like_data_row_numeric_label.csv",
        "header_none_data_only.csv",
        "quoting_backslash_escape.csv",
        "quoting_doubled_quotes.csv",
        "ragged_rows_inconsistent_columns.csv",
        "single_column.csv",
        "numeric_european_format.csv",
        "null_representations_mixed.csv",
        "gen_combo_eu_legacy.csv",
        "gen_combo_bom_crlf_preamble.csv",
    ),
}
"""Named fixture lists for ``--subset``; ``docs/evaluation.md`` explains when to use each."""

Fixtures = list[tuple[str, dict[str, Any]]]
"""``(filename, manifest entry)`` pairs, in run order."""


def select_fixtures(
    manifest: dict[str, dict[str, Any]],
    *,
    category: str | None,
    names: list[str] | None,
    max_fixtures: int | None,
) -> Fixtures:
    """The fixtures to run, sorted by name, after ``--category``/``--fixture``/``--max-fixtures``.

    Raises:
        ValueError: If a ``--fixture`` name is not in the manifest, or
            nothing is left to run.
    """
    if names:
        unknown = sorted(set(names) - manifest.keys())
        if unknown:
            raise ValueError(f"Unknown --fixture: {', '.join(unknown)}.")
    selected = [
        (filename, entry)
        for filename, entry in sorted(manifest.items())
        if (category is None or entry["category"] == category) and (not names or filename in names)
    ]
    if max_fixtures is not None:
        selected = selected[:max_fixtures]
    if not selected:
        raise ValueError("No fixture matches the filters.")
    return selected


def worst_case_calls(backend: LLMBackend, model: str, fallback_model: str) -> int:
    """The most model requests one inspection can make.

    One per candidate model (the fallback is skipped when it equals the
    primary), doubled on the cloud backend, which retries a 429/503 once.
    """
    candidates = 1 if model == fallback_model else 2
    return candidates * (2 if backend is LLMBackend.API else 1)


def calls_made(usage: Usage | None, backend: LLMBackend, model: str, fallback_model: str) -> int:
    """Model requests one inspection is charged against ``--max-calls``.

    A success reports its attempts and transient retries in ``Usage``. A
    failure reports no usage (and a failed cloud attempt hides whether it
    was retried), so it is charged the worst case, never less than it
    could have cost.
    """
    if usage is not None:
        return usage.attempts + usage.retries
    return worst_case_calls(backend, model, fallback_model)


class CallBudget:
    """The ``--max-calls`` hard stop, shared by every model of an invocation.

    A fixture only starts when its worst case still fits, so the run never
    makes more than ``max_calls`` model requests.
    """

    def __init__(self, max_calls: int | None) -> None:
        """Start with no call made; ``None`` means no limit."""
        self.max_calls = max_calls
        self.used = 0

    def allows(self, planned: int) -> bool:
        """Whether ``planned`` more calls fit in the budget."""
        return self.max_calls is None or self.used + planned <= self.max_calls

    def charge(self, calls: int) -> None:
        """Record ``calls`` model requests as made."""
        self.used += calls


class RateLimiter:
    """The ``--rpm`` guard: a sliding one-minute window of model requests.

    Before a fixture starts, it waits until the fixture's worst case fits
    in the last minute's window. Calls are stamped when the fixture ends,
    which is later than they happened, so the real rate only errs low.
    """

    WINDOW_SECONDS = 60.0

    def __init__(
        self,
        rpm: int | None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Limit to ``rpm`` requests per minute (``None``: no limit); clock and sleep for tests."""
        self.rpm = rpm
        self._clock = clock
        self._sleep = sleep
        self._stamps: list[float] = []

    def wait(self, planned: int) -> None:
        """Sleep until ``planned`` more calls stay within ``rpm`` over the last minute."""
        if self.rpm is None:
            return
        planned = min(planned, self.rpm)
        while True:
            now = self._clock()
            self._stamps = [s for s in self._stamps if s > now - self.WINDOW_SECONDS]
            if len(self._stamps) + planned <= self.rpm:
                return
            delay = self._stamps[0] + self.WINDOW_SECONDS - now
            logger.info("--rpm %d: waiting %.1fs.", self.rpm, delay)
            self._sleep(delay)

    def record(self, calls: int) -> None:
        """Stamp ``calls`` model requests as made now."""
        self._stamps.extend([self._clock()] * calls)
