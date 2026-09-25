# CLAUDE.md

Guide for Claude Code in this repo. File written caveman style on purpose. Keep under ~5 KB: loaded every turn.

## Active Skills & Behavior

- **Caveman Mode:** Active on session start. Use ultra-concise, direct language and minimal responses to save output tokens.
- Do not greet, summarize changes, or speak in long prose.
- Skill: `anthropic-skills:caveman`, level full, all chat replies.
- Commits, PRs, code comments, docstrings, repo docs (except this file): normal English prose.

## Context optimization

- Read only files task need. Module map + design decisions: `ARCHITECTURE.md` (single source; do not copy here).
- Short terminal output: `-q`, `--tb=short`, `| head`, `rg -l`, `git diff --stat`.
- Show only changed code blocks. Never rewrite whole file.
- File over 300 lines: partial read (`offset`/`limit`, targeted grep). `tests/test_csv_inspector.py` ~1300 lines.
- `agents/csv_inspector/samples/`: never read, grep, list unless strictly needed. Exclude from search (`--glob '!**/samples/**'`). Need fixture fact: read one case in `generate_samples.py`. Edit generator, never fixtures (`*.csv`/`*.tsv` byte-exact, `-text` in `.gitattributes`).
- `uv.lock` (~2000 lines): never read. Regenerate with `uv lock`.

## Overview

- Monorepo, data-engineering AI agents. Each `agents/<name>/` = self-contained pip-installable library (not service), own version, own CHANGELOG.
- Default LLM: free local Ollama. Cloud (Gemini) = opt-in `[cloud]` extra.
- One agent: `agents/csv_inspector` (`csv-inspector`). Infer encoding, dialect, header, footer, column schema of big messy CSV/TSV from small head/tail samples.
- `examples/csv_inspector_api` = FastAPI host embedding agent. Executable docs, never built/tagged/published, no CHANGELOG.
- Not on PyPI. Install from git tag `<package>-vX.Y.Z`.

## Commands

Root `pyproject.toml` = virtual uv workspace (`agents/*`, `examples/*`). `dev` group hold tools.

```bash
uv sync --all-packages --all-extras                        # install
uv run ruff check . && uv run ruff format --check .       # lint, whole repo, from root
uv run --directory agents/csv_inspector mypy               # strict
uv run --directory agents/csv_inspector pytest --cov       # floor 93%
uv run --directory examples/csv_inspector_api mypy
uv run --directory examples/csv_inspector_api pytest --cov # floor 90%
uv build --package csv-inspector                           # wheel + sdist
uv lock                                                    # after any pyproject.toml change; commit uv.lock
```

- mypy/pytest per package, never across: `conftest.py`/`fakes.py` names collide.
- Tests hermetic: no Ollama, network, credentials.
- Live run: `ollama serve` + `qwen2.5-coder:7b`, `:3b`. `uv run agents/csv_inspector/main_demo.py`, `uv run csv-inspector file.csv [--backend api]`.
- Pytest not measure LLM accuracy. Prompt/grounding change: `uv run --directory agents/csv_inspector python scripts/eval_samples.py [--category X]` before + after, both scores in PR.
- New baseline in `docs/evaluation.md`: same PR update README "Accuracy at a glance". Demo result change: rerun `scripts/render_readme_hero.py` + `docs/assets/demo.tape`.
- Dependency change: also build wheel, install in clean venv, run `scripts/smoke_test_installed.py` outside repo.

## Rules

- Cost first: no new runtime dependency, no extra model call, without asking.
- Python 3.10+ (CI 3.10–3.14 Linux, 3.14 Windows). Dev 3.14. Full type hints, Google docstrings.
- **GitHub Actions paused since 2026-09-25** (private repo, monthly minute quota exhausted): CI workflow `disabled_manually`. No PR gets CI. Run every check above locally before opening and merging a PR; say "CI paused, checks run locally" in PR body. Never re-enable (`gh workflow enable CI`) without user approval; see `CONTRIBUTING.md` "Checks".
- Public API = `csv_inspector.__all__` only (test enforce). Change `__all__` or JSON contract: ask first. Examples import `__all__` only, never `_` modules or `cli`.
- Library: module logger only. No `print()`, no `basicConfig()` outside `cli.py` (and example `main_demo.py`).
- All exceptions derive `CSVInspectorError`. No bare `except`.
- Explicit `Settings` never read env. Example builds library settings only in `ApiSettings.to_library_settings()`.
- Example `allow_backend_override` default False forever (cost guard, test check).
- User-visible change: `agents/csv_inspector/CHANGELOG.md` `[Unreleased]` (Added / Changed / Changed (breaking) / Fixed / Documentation) + README/docs in same PR. Breaking while `0.x` = minor bump.
- Branches `feat/`, `fix/`, `docs/`, `chore/`, `refactor/`. Conventional Commits. Never commit `_issues/`, `.env`, `.coverage`, `dist/`.
- New agent/example: `ARCHITECTURE.md` checklist.
