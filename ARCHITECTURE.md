# Architecture

`data-agent-toolkit` is a monorepo of **self-contained agents**. Each agent
is an independently versioned, pip-installable Python library that lives,
with everything it needs, in its own folder under `agents/`. Runnable hosts
that show how to embed an agent live under `examples/` (see
[Examples](#examples)). The repository root holds only what every agent
shares.

## Layout

```
data-agent-toolkit/
├── agents/
│   └── csv_inspector/              # One agent: the csv-inspector package
│       ├── pyproject.toml          # Package metadata + the agent's mypy/pytest/coverage config
│       ├── src/csv_inspector/      # The library (public API in __init__.py; _modules are internal)
│       ├── tests/                  # The agent's test suite (conftest.py, fakes.py, test_*.py)
│       ├── docs/                   # Guides shipped in the sdist (e.g. embedding.md)
│       ├── scripts/
│       │   ├── smoke_test_installed.py  # Post-install smoke test, run by CI
│       │   └── eval_samples.py          # Manual live-model evaluation harness
│       ├── samples/                # Generated edge-case fixtures + ground-truth manifest
│       ├── main_demo.py            # Demo: the CLI on the bundled sample.csv
│       ├── sample.csv
│       ├── .env.example            # Settings template for the agent's CLIs
│       ├── README.md               # Package README (also the PyPI long description)
│       ├── CHANGELOG.md
│       └── LICENSE                 # Copy of the repository license, shipped in the dists
├── examples/
│   └── csv_inspector_api/          # One example: a FastAPI host embedding csv-inspector
│       ├── pyproject.toml          # [project] + dependencies only (never built) + mypy/pytest/coverage config
│       ├── src/csv_inspector_api/  # The host application
│       ├── tests/                  # Hermetic tests (fake model invoker, no network)
│       ├── main_demo.py
│       ├── .env.example
│       └── README.md
├── .github/
│   ├── workflows/ci.yml            # Discovers the agents and runs every job per agent
│   ├── ISSUE_TEMPLATE/, pull_request_template.md, dependabot.yml
├── pyproject.toml                  # uv workspace, dev tooling group, shared ruff defaults
├── uv.lock                         # Locked development environment (all agents + tooling)
├── .python-version                 # Development Python (3.14)
├── .pre-commit-config.yaml
├── README.md, ARCHITECTURE.md, CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md
└── LICENSE
```

## What lives where

| Concern | Repository root | `agents/<name>/` |
|---|---|---|
| Code, tests, fixtures, demo, scripts | — | ✅ everything |
| Package metadata, dependencies, extras | — | `pyproject.toml` `[project]` |
| Lint and format rules | shared defaults (`[tool.ruff]`) | `extend`s the root, adds its own ignores |
| Type-checking, test runner, coverage | — | `[tool.mypy]`, `[tool.pytest.ini_options]`, `[tool.coverage.*]` |
| Development environment | uv workspace + `dev` group, `uv.lock` | a workspace member by existing |
| CI | one workflow, parameterised by agent | provides `scripts/smoke_test_installed.py` |
| Docs | index README, contribution and policy docs | README (PyPI page), CHANGELOG, `docs/` |
| License | `LICENSE` (MIT, the whole repository) | identical copy, shipped in its dists |

### Why per-agent tooling runs

mypy and pytest run **once per agent, from the agent's folder**, with that
agent's configuration. This is necessary, not cosmetic. Every agent has
its own `tests/conftest.py` and test helpers (such as `fakes.py`), and one
run over several agents would collide on those module names. Per-agent
runs also keep one agent's settings (strictness exceptions, fixture paths,
coverage floor) from leaking into another.

ruff runs once over the whole repository. It resolves the closest
`pyproject.toml` for each file, so every agent's `extend` of the root
defaults applies automatically.

### Development environment

The root `pyproject.toml` is a *virtual* [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/)
(it has no `[project]` table and is never built). Its members are
`agents/*` and `examples/*`, so a new agent or example joins the workspace
just by existing. `uv sync --all-packages` installs every agent in editable
mode, with its extras, every example, plus the `dev` tooling group, all
pinned in `uv.lock`. Tests therefore import each package
exactly as an external host would (`import csv_inspector`), never through
`sys.path` tricks.

### CI

[`ci.yml`](.github/workflows/ci.yml) first discovers the agents from
`agents/*/pyproject.toml`, together with their extras. It then runs these
jobs:

- **Lint & lockfile** (once): ruff, and `uv sync --locked`, which fails if
  `uv.lock` is stale.
- **Type-check** (per agent): `mypy --strict`.
- **Tests** (per agent × Python 3.10–3.14 on Linux, plus 3.14 on Windows):
  `pytest --cov` with branch coverage and the agent's `fail_under` floor.
- **Tests, lowest dependencies** (per agent, Python 3.10): installs the
  lowest versions the agent's ranges allow for its direct dependencies
  (`uv pip install --resolution lowest-direct`, every extra) and runs
  `pytest`. The other jobs test `uv.lock`, so only this one catches a floor
  that is too low.
- **Package** (per agent × Python 3.10 and 3.14): builds the sdist and
  wheel, runs `twine check --strict`, and installs with **plain pip** into
  fresh virtual environments: the wheel without extras, and the sdist with
  every extra (`pip check` included). It then runs the agent's
  `scripts/smoke_test_installed.py` from outside the repository against both.

The examples are discovered the same way, from `examples/*/pyproject.toml`,
and each one gets two lighter jobs: **Type-check example** (`mypy`) and
**Tests example** (`pytest --cov` with the example's `fail_under`, plus a
coverage summary). Both run on Python 3.14 on Linux only. Examples are not
shipped, so there is no Windows leg, no lowest-dependencies job and no
packaging job. Ruff and the lockfile check already cover them through the
**Lint & lockfile** job. The examples jobs run on every pull request, not
only on changes under `examples/`, because an example tests the current
source of the agent it embeds. With no example, the jobs are skipped
instead of failing on an empty matrix. One more job, **Docker image
(csv_inspector_api)**, builds that example's image from the repository root,
checks its size and that `GET /health` answers from a running container; it
never pushes to a registry.

## Examples

An example under `examples/<name>/` is a small, runnable host that embeds an
agent: an HTTP API today, perhaps a worker or a notebook later. It is
**executable documentation** of how to embed an agent, meant to be read and
copied, not deployed as is.

An example is **not** a distribution:

- no `[build-system]` and no `[project.scripts]`: it is never built, so it
  has no wheel, sdist or console script;
- no release tag, no PyPI upload, no SemVer contract and no `CHANGELOG.md`:
  it may change in any commit, and its changes are tracked through pull
  requests only (the repository has no root `CHANGELOG.md` either);
- no `scripts/smoke_test_installed.py`: there is no installed artefact to
  smoke-test.

It still has a `[project]` table (name, `version = "0.0.0"`,
`requires-python`, dependencies), because that is how a uv workspace member
declares its dependencies.

**Depends on the agent's source.** An example is a workspace member and
takes the agent it embeds as a workspace source:

```toml
[tool.uv.sources]
csv-inspector = { workspace = true }
```

It therefore always runs against the agent's current source, never against
a tag, and a change to an agent that breaks one of its examples fails CI in
the same pull request. Like any host, an example imports only the agent's
public API (its `__all__`), never its `_` modules.

**Tooling.** Same as an agent: `[tool.ruff] extend = "../../pyproject.toml"`
for the shared lint and format rules, plus its own `[tool.mypy]`,
`[tool.pytest.ini_options]` and `[tool.coverage.*]`, run from the example's
folder. Its tests are hermetic like the agents' ones: every model backend is
faked through the agent's injection points, with no Ollama, no credentials
and no network.

**CI.** Examples get a lighter job set than agents: type-check and tests on
a single Python version on Linux, and no lowest-dependencies or packaging
jobs, since nothing is shipped. Lint and the lockfile check cover them
through the root jobs. An example with a `Dockerfile` gets a build-and-smoke
job, never a push.

## Policies

- **License.** The whole repository is MIT-licensed (`LICENSE`). Each agent
  carries an identical copy because a distribution can only include files
  from inside its own project folder.
- **Python versions.** Libraries support **Python 3.10+**
  (`requires-python`), because a host should not have to upgrade Python to
  embed an agent, and CI tests every version in that range. Development and
  tooling run on the newest Python (`.python-version`: 3.14). Raising an
  agent's floor is a breaking change, recorded in its CHANGELOG.
- **Versioning and releases.** Each agent follows [SemVer](https://semver.org/)
  on its own and is released by tagging `<package>-vX.Y.Z` (for example,
  `csv-inspector-v0.1.0`). Hosts install from the tag
  (`pip install "csv-inspector @ git+...@csv-inspector-v0.1.0#subdirectory=agents/csv_inspector"`).
- **PyPI.** Agents are **not published to PyPI yet**: the APIs are still
  `0.x`, and the git tags already give reproducible installs. When an agent
  is published, it will be through PyPI Trusted Publishing (OIDC from a
  release workflow, no long-lived tokens). The CI packaging job already
  guarantees the artefacts are uploadable.
- **Dependencies.** Each agent declares version *ranges* in its
  `pyproject.toml`. Those ranges are a contract with its hosts and are only
  changed deliberately. CI tests both ends of each range: `uv.lock` for the
  newest versions and the lowest-dependencies job for the floors. The
  development environment is pinned in
  `uv.lock`, which Dependabot refreshes weekly, as it does the GitHub Actions.

## Adding a new agent

1. Create `agents/<agent_name>/` following the layout above:
   `pyproject.toml`, `src/<package>/` with a `py.typed` marker and an
   explicit `__all__` in `__init__.py` (internal modules prefixed with `_`),
   `tests/`, `scripts/smoke_test_installed.py`, `README.md`, `CHANGELOG.md`
   and a copy of `LICENSE`.
2. In its `pyproject.toml`, add `[tool.ruff] extend = "../../pyproject.toml"`
   and its own `[tool.mypy]`, `[tool.pytest.ini_options]` and
   `[tool.coverage.*]` sections (use `csv_inspector`'s as the template).
3. Default to local, free execution via Ollama. Make cloud/API backends an
   opt-in extra with lazily imported SDKs, and accept injected settings.
4. Return Pydantic-validated structured output, never free text.
5. Fake every external backend in its tests, so the suite runs without
   credentials or network access.
6. Run `uv lock`, and add a `mypy-<agent>` hook to `.pre-commit-config.yaml`.
7. Add it to the agents table in the root `README.md` and to `SECURITY.md`.

CI and the workspace pick the new agent up on their own; nothing else at the
root needs to change.

## Adding a new example

1. Create `examples/<example_name>/` following the layout above:
   `pyproject.toml`, `src/<package>/` with a `py.typed` marker and an
   explicit `__all__` in `__init__.py`, `tests/`, `main_demo.py`,
   `.env.example` and `README.md`. No `CHANGELOG.md`, `LICENSE` copy or
   `scripts/smoke_test_installed.py`.
2. In its `pyproject.toml`, declare `[project]` with `version = "0.0.0"`
   and its dependencies, but **no `[build-system]` and no
   `[project.scripts]`**. Take the agent it embeds as a workspace source
   (`[tool.uv.sources] <agent-package> = { workspace = true }`), and keep
   test-only tools in a local `[dependency-groups] dev`.
3. Add `[tool.ruff] extend = "../../pyproject.toml"` and its own
   `[tool.mypy]`, `[tool.pytest.ini_options]` and `[tool.coverage.*]`
   sections (use `csv_inspector_api`'s as the template).
4. Import only the agent's public API, pass it injected settings built in
   one place (never the agent's own environment loading), and default to
   the free local Ollama backend.
5. Fake every model backend in its tests, so the suite runs without
   credentials or network access.
6. Run `uv lock`, and add a `mypy-<example>` hook to `.pre-commit-config.yaml`.
7. Add it to the examples table in the root `README.md`. Examples are out
   of scope for `SECURITY.md`: do not add them there.

CI and the workspace pick the new example up on their own; nothing else at
the root needs to change.
