# CLAUDE.md

Guide for Claude Code in this repo. File written caveman style on purpose.

## Active Skills & Behavior

- **Caveman Mode:** Active on session start. Use ultra-concise, direct language and minimal responses to save output tokens.
- Do not greet, summarize changes, or speak in long prose.
- Skill: `anthropic-skills:caveman`, level full, all chat replies.
- Commits, PRs, code comments, docstrings, repo docs (except this file): normal English prose.

## Context optimization

- Read only files task need.
- Short terminal output: `-q`, `--tb=short`, `| head`, `rg -l`, `git diff --stat`.
- Show only changed code blocks. Never rewrite whole file.
- File over 300 lines: partial read (`offset`/`limit`, targeted grep).
- `agents/csv_inspector/samples/`: never read, grep, list unless strictly needed. Exclude from search (`--glob '!**/samples/**'`). Need fixture fact: read one case in `generate_samples.py`.
- `uv.lock` (~2000 lines): never read. Regenerate with `uv lock`.

## Overview

- Monorepo, data-engineering AI agents. Each `agents/<name>/` = self-contained pip-installable library (not service), own version.
- Default LLM: free local Ollama. Cloud (Gemini) = opt-in `[cloud]` extra.
- One agent now: `agents/csv_inspector` (package `csv-inspector`). Infer encoding, dialect, header row, footer, column schema of big messy CSV/TSV from small head/tail samples.
- Not on PyPI. Install from git tag.

## Commands

Root `pyproject.toml` = virtual uv workspace (members `agents/*`, no `[project]`). `dev` group hold tools.

```bash
uv sync --all-packages --all-extras                       # install
uv run ruff check . && uv run ruff format --check .      # lint, root, whole repo
uv build --package csv-inspector                          # build wheel + sdist into dist/
uv run pre-commit install                                 # git hooks
cd agents/csv_inspector
uv run mypy                                               # strict + pydantic plugin
uv run pytest --cov                                       # branch coverage, floor 93%
uv run pytest tests/test_csv_inspector.py::test_name      # single test
uv lock                                                   # after any pyproject.toml change; commit uv.lock
```

- Run mypy/pytest per agent, from agent folder (or `uv run --directory agents/<name> ...`). Never across agents: `conftest.py`/`fakes.py` names collide.
- Tests hermetic: no Ollama, network, credentials.
- Live run need `ollama serve` + `qwen2.5-coder:7b`, `qwen2.5-coder:3b`: `uv run agents/csv_inspector/main_demo.py`, `uv run csv-inspector file.csv [--backend api]`.
- Pytest not measure LLM accuracy. Prompt change: run `uv run agents/csv_inspector/scripts/eval_samples.py` before + after, both scores in PR.

## Root files

- `pyproject.toml`: uv workspace + `dev` group (build, mypy, pre-commit, pytest, pytest-cov, ruff, twine). Shared ruff config: `py310`, 100 cols, rules E W F I N D UP ANN B BLE C4 SIM RET PTH PT PL RUF, Google docstrings.
- `uv.lock`: workspace lockfile. Commit with every dependency change.
- `.python-version`: `3.14` (dev version).
- `.pre-commit-config.yaml`: local hooks `uv-lock`, `ruff-check`, `ruff-format`, `mypy-csv-inspector`.
- `.gitattributes`: all text LF. `*.csv`, `*.tsv` = `-text` (byte-exact, git never rewrite).
- `.gitignore`: `.venv`, caches, coverage, `dist/`, `.env`.
- `README.md`: repo intro, agents list, design principles, layout, dev setup.
- `ARCHITECTURE.md`: layout rationale, what live where, why per-agent tooling, CI design, policies, new-agent checklist.
- `CONTRIBUTING.md`: setup, checks, code standards, PR rules, release process.
- `SECURITY.md`: supported versions, private vuln report, scope.
- `CODE_OF_CONDUCT.md`, `LICENSE` (MIT).

## .github/

- `workflows/ci.yml`: jobs `agents` (discover `agents/*`), `lint` (`uv sync --locked` + ruff check/format), `typecheck` (mypy per agent), `test` (pytest --cov, Python 3.10–3.14 Linux + 3.14 Windows, coverage summary), `test-lowest` (Python 3.10, `--resolution lowest-direct`, all extras, pytest: catch too-low floors), `package` (`python -m build`, `twine check --strict`, install wheel and sdist in clean venvs, run `smoke_test_installed.py` outside repo).
- `dependabot.yml`: weekly updates, `github-actions` + `uv`.
- `pull_request_template.md`: PR checklist.
- `ISSUE_TEMPLATE/`: `bug_report.yml`, `feature_request.yml`, `config.yml` (blank issues on, private security-report link).

## agents/csv_inspector/ (package root)

- `pyproject.toml`: hatchling build. Python `>=3.10`. Deps `chardet>=5.2,<8`, `ollama>=0.6.2,<1`, `pydantic>=2.6,<3`. Extra `[cloud]`: `google-genai>=2.0,<3`, `pydantic-settings>=2.0,<3`. Script `csv-inspector = csv_inspector.cli:main`. Sdist ship only `src`, `docs`, README, CHANGELOG, LICENSE. Mypy strict on src, scripts, samples, main_demo, tests. Coverage `fail_under = 93`. Ranges = host contract, change on purpose only.
- `README.md`: install, quickstart, public API, sources, timeouts, how it work, backends, settings, CLI, errors.
- `CHANGELOG.md`: Keep a Changelog. User-visible change go to `[Unreleased]`.
- `.env.example`: env vars `LLM_BACKEND`, `OLLAMA_MODEL`, `OLLAMA_FALLBACK_MODEL`, `GEMINI_API_KEY`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, `CLOUD_MODEL`, `CLOUD_FALLBACK_MODEL`.
- `main_demo.py`: thin wrapper over CLI, default file `sample.csv`.
- `sample.csv`: 8-line demo CSV for `main_demo.py`.
- `docs/embedding.md`: host-app guide. Dependency, pass existing data, sync hosts (Flask, Django), async hosts (FastAPI), config + secrets, timeouts + errors, thread-safety, logging, non-goals.
- `docs/using-the-result.md`: result to reader options. stdlib `csv` recipe (test run it), pandas (`skiprows` not `header`, `skipfooter` python engine), PySpark (option map, drop preamble/footer lines), encoding names Spark/BigQuery.
- `LICENSE`: MIT.

## src/csv_inspector/ (library)

Public API = `__all__` only (test enforce). `_`-modules internal.

- `__init__.py`: export `inspect_csv`, `ainspect_csv`, `CSVSource`, `CSVInspectionResult`, `ColumnSchema`, `ColumnType`, `LLMBackend`, `Settings`, `load_settings`, all exceptions, `__version__`. Package logger `NullHandler`.
- `__main__.py`: `python -m csv_inspector`, call `cli.main`.
- `py.typed`: PEP 561 marker.
- `_backends.py`: `LLMBackend` enum `LOCAL="local"`, `API="api"`. No third-party imports.
- `_config.py`: frozen Pydantic `Settings` (`llm_backend`, `ollama_model`, `ollama_fallback_model`, `gemini_api_key` SecretStr, `google_cloud_project`, `google_cloud_location`, `cloud_model`, `cloud_fallback_model`). Defaults `qwen2.5-coder:7b`/`:3b`, `gemini-2.5-flash`/`-flash-lite`. `Settings()` never read env. `load_settings(env_file=)` read env, need `[cloud]`. `CloudAuthMode` `gemini_api`/`vertex_ai`, `CloudCredentials`. `resolve_settings`, `ensure_backend_ready` check backend **before** source read (non-seekable stream never wasted).
- `_sampling.py`: bounded head read (4 KiB default) + tail of uncovered bytes only. Each window max 16 KiB. Readers: path, bytes/buffer, seekable stream, non-seekable forward stream (64 KiB chunks, max forward scan 64 MiB). Text stream rejected. `detect_encoding` (chardet), `decode_sample`. Tail decoded BOM-less, code-unit aligned. Negative `read()` = read all, budgets validated. Empty source: `EmptySampleError`, no model call. `Samples` dataclass, `sample_source`, `describe_source`.
- `_prompt.py`: `SYSTEM_PROMPT` (JSON only). `build_prompt` (head, or head + tail with mid-line caveat). `parse_and_validate`: lenient JSON extract (fenced or bare), validate into `CSVInspectionResult`.
- `_models.py`: `ColumnType` Literal (string, integer, float, date, datetime, boolean; aliases mapped, unknown = string). `ColumnSchema` (`name`, `inferred_type`, `nullable`, `example_values`). `CSVInspectionResult` (`encoding`, `delimiter`, `quotechar`, `escapechar`, `doublequote`, `header_row_index`, `footer_lines`, `columns`, `confidence` 0–1, `notes`). Normalize `\t`/`tab`, null escape spellings.
- `_invokers.py`: sync/async Ollama + Gemini invokers. New client per call, closed after (thread-safe). Lazy SDK import. Ollama `num_ctx` sized to prompt (4096–32768, 1024 response tokens). Secrets redacted in errors. `builtin_invoker`, `builtin_async_invoker` pick by backend.
- `_inspect.py`: `inspect_csv(source, *, backend, settings, model, fallback_model, n_bytes, tail_bytes, timeout_seconds, model_invoker)`, `ainspect_csv` mirror (sampling in worker thread). Primary then fallback model. `timeout_seconds` = one budget whole model phase; each model get equal share of remainder; enforced for custom `model_invoker` too. `BackendConfigurationError` skip fallback.
- `_grounding.py`: `ground_in_samples`. Small models miscount/paraphrase. Model answer = key to recompute `header_row_index`, literal column names, verbatim footer from samples. Delimiter absent from head: best of `,` `;` tab `|` by field-count agreement. Unanchored footer dropped when end sampled. Totals label regex (EN/ES: total, totales, suma, sum, subtotal, grand total). Never promote unlabelled data row to footer.
- `_exceptions.py`: base `CSVInspectorError`. Children `FileSampleReadError`, `EmptySampleError`, `ModelInvocationError` (> `BackendConfigurationError` > `CredentialsNotConfiguredError`; > `ModelTimeoutError`), `ResponseParsingError`, `SchemaValidationError`, `InspectionFailedError` (> `InspectionTimeoutError`).
- `cli.py`: only module with `print()`, `logging.basicConfig()`, `.env` read. Args: `file`, `--backend`, `--model`, `--fallback-model`, `--bytes`, `--tail-bytes`, `--timeout` (default 300 s, 0 = none), `--env-file`, `--no-env-file`, `--log-level`. Print result as indented JSON.

## scripts/

- `eval_samples.py`: manual LLM accuracy harness. Real Ollama vs every `samples/manifest.json` case, per-field score report. `--timeout` per fixture (default 300 s, 0 = none), timeout = errored fixture. Not pytest, not CI.
- `smoke_test_installed.py`: CI run on installed wheel/sdist outside repo. Fake invoker, check public API end to end.

## tests/

- `conftest.py`: autouse fixture clear settings env vars, `chdir` to empty tmp dir.
- `fakes.py`: fake backends/invokers, no network ever.
- `test_csv_inspector.py` (69 tests, ~1200 lines, partial read): core pipeline, prompt, parsing, grounding, fallback.
- `test_sources.py`: path, bytes, seekable + non-seekable streams.
- `test_timeouts_and_async.py`: time budget sync/async, `ainspect_csv`.
- `test_backends.py`: backend selection, Gemini invoker.
- `test_config.py`: `Settings`, `load_settings`, env handling.
- `test_cli_support.py`: CLI helpers, arg parsing.
- `test_embedding.py`: `__all__` surface, no `print`/`basicConfig` outside `cli.py` (AST), `NullHandler`, isolation.
- `test_samples_catalog.py`: LLM-free checks over fixtures; fixtures byte-match generator.
- `test_eval_samples.py`: scoring logic of `eval_samples.py`.
- `test_docs_recipes.py`: exec stdlib recipe from `docs/using-the-result.md` vs fixtures.
- Package imported as installed. Only `samples/`, `scripts/` on `sys.path`.

## samples/ (do not open)

- Generated byte-exact CSV/TSV fixtures + `manifest.json`, by `uv run agents/csv_inspector/samples/generate_samples.py`. Edit generator, never fixtures.

## Rules

- Python 3.10+ (CI 3.10–3.14 Linux, 3.14 Windows). Dev 3.14.
- Library: module logger only. No `print()`, no `basicConfig()` outside `cli.py`.
- All exceptions derive `CSVInspectorError`.
- Branches `feat/`, `fix/`, `docs/`, `chore/`. Release tag `<package>-vX.Y.Z`.
- New agent: follow `ARCHITECTURE.md` checklist.
