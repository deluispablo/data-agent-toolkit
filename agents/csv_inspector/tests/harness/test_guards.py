"""Tests of ``eval_harness.guards``: fixture selection and the quota guards."""

from __future__ import annotations

import json

import pytest

from eval_harness.guards import MANIFEST_PATH, SUBSETS, CallBudget, RateLimiter, select_fixtures


def test_call_budget_admits_a_fixture_only_when_its_worst_case_fits() -> None:
    """The budget never lets the run exceed ``max_calls``; ``None`` is unlimited."""
    budget = CallBudget(3)
    budget.charge(2)

    assert budget.allows(1)
    assert not budget.allows(2)
    assert CallBudget(None).allows(10**6)


def test_rate_limiter_waits_for_the_window_to_free_up() -> None:
    """With the window full it sleeps until the oldest call is a minute old."""
    now = [0.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    limiter = RateLimiter(2, clock=lambda: now[0], sleep=sleep)
    limiter.wait(2)
    limiter.record(2)
    now[0] = 10.0
    limiter.wait(5)  # capped at rpm: waits for the whole window

    assert sleeps == [pytest.approx(50.0)]
    RateLimiter(None, sleep=sleep).wait(100)
    assert len(sleeps) == 1


def test_select_fixtures_filters_and_caps() -> None:
    """``--fixture``, ``--category`` and ``--max-fixtures`` narrow the sorted catalog."""
    manifest = {
        "b.csv": {"category": "x"},
        "a.csv": {"category": "x"},
        "c.csv": {"category": "y"},
    }

    assert [f for f, _ in select_fixtures(manifest, category="x", names=None, max_fixtures=1)] == [
        "a.csv"
    ]
    assert [
        f for f, _ in select_fixtures(manifest, category=None, names=["c.csv"], max_fixtures=None)
    ] == ["c.csv"]
    with pytest.raises(ValueError, match=r"Unknown --fixture: z.csv"):
        select_fixtures(manifest, category=None, names=["z.csv"], max_fixtures=None)
    with pytest.raises(ValueError, match="No fixture"):
        select_fixtures(manifest, category="none", names=None, max_fixtures=None)


def test_subsets_name_manifest_fixtures() -> None:
    """The quick subset covers every category; cloud is the documented 15 regular fixtures."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for names in SUBSETS.values():
        assert set(names) <= manifest.keys()
        assert len(set(names)) == len(names)
    assert {manifest[name]["category"] for name in SUBSETS["quick"]} == {
        entry["category"] for entry in manifest.values()
    }
    assert len(SUBSETS["cloud"]) == 15
    assert not any(manifest[name]["known_limitation"] for name in SUBSETS["cloud"])
