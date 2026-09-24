# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Style

Use caveman skill (`anthropic-skills:caveman`, level full) for all chat replies, every session. Commits, PRs, code comments, docstrings, repo docs: normal English prose.

## Token budget

Never read, grep or list `agents/csv_inspector/samples/` (test CSV/TSV fixtures, `manifest.json`, `generate_samples.py`) unless task strictly needs it. Exclude it from searches (e.g. `--glob '!**/samples/**'`). Need fixture facts: read one targeted case in `generate_samples.py`, not whole folder.

## Overview

Monorepo of data-engineering AI agents. Each `agents/<name>/` = self-contained, pip-installable library (not service), own version. Default LLM: free local Ollama. Cloud (Gemini) = opt-in `[cloud]` extra. One agent now: `agents/csv_inspector` (package `csv-inspector`). Layout rationale + new-agent checklist: `ARCHITECTURE.md`.

## Commands

Root `pyproject.toml` = virtual uv workspace (members `agents/*`, no `[project]`), `dev` group holds tools.

```bash
uv sync --all-packages --all-extras
uv run ruff check . && uv run ruff format --check .      # root, whole repo
cd agents/csv_inspector
uv run mypy                                               # strict + pydantic plugin
uv run pytest --cov                                       # branch coverage, floor 93%
uv run pytest tests/test_csv_inspector.py::test_name      # single test
uv lock                                                   # after any pyproject.toml change; commit uv.lock
```

- Run mypy/pytest per agent, from agent folder (or `uv run --directory agents/<name> ...`). Never across agents: `conftest.py`/`fakes.py` names collide.
- Tests hermetic: no Ollama, network, credentials.
- Live runs need `ollama serve` + `qwen2.5-coder:7b` and `qwen2.5-coder:3b`: `uv run agents/csv_inspector/main_demo.py`, `uv run csv-inspector file.csv [--backend api]`.
- Pytest not measure LLM accuracy. Prompt change: run `uv run agents/csv_inspector/scripts/eval_samples.py` before + after, put both scores in PR.

## csv_inspector pipeline (`_inspect.py` orchestrates)

Public API = `csv_inspector.__all__` only (test enforces). `_`-modules internal.

1. `_config.py`: resolve `Settings`, check backend ready **before** reading source (non-seekable stream never wasted). `Settings()` never reads env; `load_settings()` does, explicitly, needs `[cloud]`.
2. `_sampling.py`: bounded head read (4 KiB default) + tail of uncovered bytes only (each window max 16 KiB). Readers for path, bytes, seekable and non-seekable streams. chardet encoding. Tail decoded BOM-less, code-unit aligned. Byte budgets validated (negative `read()` = read all). Empty source: `EmptySampleError`, no model call.
3. `_prompt.py`: build prompt (head, or head + tail with mid-line caveat); lenient JSON extract; validate into `CSVInspectionResult` (`_models.py`).
4. `_invokers.py`: sync/async Ollama + Gemini. New client per call (thread-safe). Lazy SDK imports. Ollama `num_ctx` sized to prompt. Secrets redacted in errors.
5. `_inspect.py`: primary then fallback model. `timeout_seconds` = one budget for whole model phase; each model gets equal share of remainder; library enforces it, custom `model_invoker` too. `BackendConfigurationError` skips fallback. `ainspect_csv` mirrors sync, sampling in worker thread.
6. `_grounding.py`: small models miscount/paraphrase. Model answer used as key to recompute `header_row_index`, literal column names, verbatim footer from samples. Never promote unlabelled data row to footer.

All exceptions derive `CSVInspectorError` (`_exceptions.py`).

## Enforced rules

- `print()` / `logging.basicConfig()` only in `cli.py` (AST test). Library: module logger, package `NullHandler` only.
- Python 3.10+ (CI: 3.10–3.14 Linux, 3.14 Windows). Dev on 3.14. Ruff `py310`, 100 cols, Google docstrings.
- `conftest.py` autouse fixture clears settings env vars, `chdir` to empty tmp dir. Backends faked in `tests/fakes.py`. Package imported as installed; only `samples/`, `scripts/` on `sys.path`.
- `samples/*.csv|tsv` byte-exact (`-text` in `.gitattributes`). Generated with `manifest.json` by `uv run agents/csv_inspector/samples/generate_samples.py`. Edit generator, never fixtures; `test_samples_catalog.py` checks byte match.
- CI `package` job: build wheel + sdist, plain-pip install in clean venvs, run `scripts/smoke_test_installed.py` outside repo. Sdist ships only `src`, `docs`, README, CHANGELOG, LICENSE.
- Dependency ranges in agent `pyproject.toml` = host contract. Change on purpose only, commit `uv.lock` same PR.
- User-visible change: `[Unreleased]` in agent `CHANGELOG.md`. Branches `feat/`, `fix/`, `docs/`, `chore/`. Release tag `<package>-vX.Y.Z`. Not on PyPI.
