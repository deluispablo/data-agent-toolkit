# Contributing

Thanks for your interest in `data-agent-toolkit`! Bug reports, fixes,
documentation and new agents are all welcome. By taking part you agree to
follow the [Code of Conduct](CODE_OF_CONDUCT.md). For security issues, see
[SECURITY.md](SECURITY.md) instead of opening an issue.

## Before you start

- **Open an issue first** for anything beyond a small fix, using the
  [issue templates](https://github.com/deluispablo/data-agent-toolkit/issues/new/choose),
  so the approach can be agreed before you invest time in it.
- Read [ARCHITECTURE.md](ARCHITECTURE.md): what lives at the repository
  root, what lives in each agent, and why.
- Changes must respect the [design principles](README.md#design-principles).
  In particular, local and free execution is the default, reads and model
  calls stay bounded, and output is validated and structured.

## Development setup

You need [uv](https://docs.astral.sh/uv/getting-started/installation/). It
installs the development Python (3.14, from `.python-version`) if you do not
have it.

```bash
git clone https://github.com/deluispablo/data-agent-toolkit
cd data-agent-toolkit
uv sync --all-packages --all-extras   # every agent (editable, with extras) + dev tools, from uv.lock
uv run pre-commit install             # run the checks on every commit
```

Only live runs need Ollama, never the tests:

```bash
ollama serve
ollama pull qwen2.5-coder:7b
ollama pull qwen2.5-coder:3b          # default fallback model

uv run agents/csv_inspector/main_demo.py            # or: uv run csv-inspector path/to/file.csv
uv run agents/csv_inspector/scripts/eval_samples.py # score a live model on the fixture catalog
```

The CLIs read their settings from environment variables and, as
applications, from `./.env` when present (copy the agent's `.env.example`).
The libraries themselves never read a `.env` file implicitly.

## Checks

CI runs these checks on every pull request and on every push to `main`;
run them locally before opening a PR too. `pre-commit` runs ruff, mypy and
the lockfile check on each commit.

```bash
uv run ruff check .                 # lint (whole repository)
uv run ruff format --check .        # formatting (whole repository)

cd agents/csv_inspector             # then, per agent:
uv run mypy                         # strict static type-check (with the pydantic plugin)
uv run pytest --cov                 # tests + branch coverage (fails under the agent's floor)
```

Examples under `examples/` have the same per-folder checks, run the same
way (see [ARCHITECTURE.md](ARCHITECTURE.md#examples)):

```bash
uv run --directory examples/<name> mypy
uv run --directory examples/<name> pytest --cov
```

A change to an agent must keep the examples that embed it green, since they
run against the agent's current source.

The test suites are **hermetic**: no Ollama, no credentials, no network.
Every model backend is faked. In `csv_inspector`, Ollama is replaced by a
stand-in module, and the Gemini client by a recorder that keeps the SDK's
real request types. An autouse fixture clears every settings variable and
runs each test from an empty directory. The one thing `pytest` deliberately
does not cover is the LLM's accuracy. That is non-deterministic, so it is
measured manually with the agent's `scripts/eval_samples.py` against a live
model. Mention the before and after scores in the PR when you change a
prompt.

## Code standards

- 100% English source code, comments and docstrings.
- [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
  on every public module, class and function, with arguments, return values
  and raised exceptions documented explicitly.
- Full static type hints (PEP 604 unions), checked with `mypy --strict`.
  Packages ship `py.typed`.
- PEP 8 and PEP 257, with formatting enforced by ruff (lint and format,
  100-column lines).
- **Library code never configures logging or prints.** It logs under its
  package logger with only a `NullHandler`. `logging.basicConfig()` and
  `print()` belong in the CLI layer only (enforced by a test).
- Explicit, typed, domain-specific exception hierarchies instead of bare
  `Exception` handling.
- The public API is exactly the package's `__all__` (enforced by a test).
  Everything else is internal and may change without notice.
- Code must run on the agent's whole supported Python range (3.10+), not
  only on the development Python.

## Pull requests

- Branch from `main` with a descriptive prefix: `feat/`, `fix/`, `docs/`
  or `chore/`.
- Keep a pull request to one purpose, and each commit to one coherent step
  with an imperative subject (for example, "Detect footers from the tail
  sample"). Reference the issue (`Refs #12`, or `Closes #12` in the PR).
- Add tests for every behaviour change. A bug fix starts with a test that
  fails without it.
- Record user-visible changes under `[Unreleased]` in the agent's
  `CHANGELOG.md` ([Keep a Changelog](https://keepachangelog.com/en/1.1.0/)).
  List breaking changes under **Changed (breaking)**.
- Dependencies: change an agent's version *ranges* in its `pyproject.toml`
  only on purpose, since they are a contract with its hosts. Then run
  `uv lock` and commit `uv.lock` in the same pull request.
- Documentation: the module map and the design decisions live only in
  `ARCHITECTURE.md`, and design notes live in module docstrings. Keep
  `CLAUDE.md` (loaded into every AI-assistant session) under about 5 KB,
  and an example's README under about 10 KB.

## Releases

Maintainers release an agent by moving its `[Unreleased]` changelog entries
under the new version, bumping `version` in its `pyproject.toml`, and
tagging `<package>-vX.Y.Z`. See [ARCHITECTURE.md](ARCHITECTURE.md#policies)
for the versioning and publishing policy.
