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
│       │   ├── eval_samples.py          # Manual live-model evaluation harness (JSONL runs)
│       │   └── compare_runs.py          # Markdown comparison of two or more eval runs
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

**Paused since 2026-09-25**: the workflow is disabled by hand because the
private repository's monthly Actions quota ran out; checks run locally
until the maintainer re-enables it (see `CONTRIBUTING.md`, "Checks").

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

## Module map

This is the one place that lists what each module does. `CLAUDE.md` and
the READMEs point here instead of repeating it.

### `agents/csv_inspector` (package `csv-inspector`)

The public API is exactly `csv_inspector.__all__`, and a test enforces it.
Modules prefixed with `_` are internal.

| Module | Responsibility |
|---|---|
| `__init__.py` | The public exports (`inspect_csv`, `ainspect_csv`, `ensure_backend_ready`, `CSVSource`, `CSVInspectionResult`, `Usage`, `LLMBackend`, `Settings`, `load_settings`, `DEFAULT_SAMPLE_BYTES`, `DEFAULT_TAIL_BYTES`, `MAX_SAMPLE_BYTES`, `ModelInvoker`, `AsyncModelInvoker`, every exception, `__version__`) and the package logger's `NullHandler`. |
| `_backends.py` | `LLMBackend` (`local`, `api`); no third-party imports. |
| `_config.py` | The frozen `Settings` model, which never reads the environment when constructed. `load_settings(env_file=None)` reads the environment and an optional `.env` with the standard library. `resolve_settings`, `CloudCredentials`, and `ensure_backend_ready`, which checks a backend before any source is read. |
| `_sampling.py` | Bounded I/O: the head window (4 KiB by default) and a tail of the bytes the head did not cover (each window at most 16 KiB), from a path, a buffer, a seekable stream or a forward-only stream (64 KiB chunks, at most 64 MiB scanned past the head). A truncated head is trimmed to its last line break. The texts handed on keep at most `MAX_HEAD_LINES` (15) head and `MAX_TAIL_LINES` (10) tail lines; a whole file longer than both gets a tail synthesized from its last lines. An empty source raises `EmptySampleError` before any model call. |
| `_encoding.py` | Encoding detection (chardet) and decoding; BOM-less, code-unit-aligned tail codecs; the shared line-break pattern and `split_lines`. No I/O. |
| `_prompt.py` | The system prompt, `build_prompt` (the fields' semantics, no JSON shape), `response_schema()` (the cached, annotation-free JSON Schema of `_ModelAnswer` both backends send), and `parse_and_validate`: lenient JSON extraction (fenced or bare) into a `_ModelAnswer`, for built-in and custom invokers alike. |
| `_models.py` | What the model answers vs what the library returns. The private `_ModelAnswer` is the answer (one footer anchor line, `footer_first_line`), with one lenient pass that normalizes small-model slips (tab, "no escape", "no quoting", `null`/`-1` header index, escaped quote, percent confidence, multi-line anchor) and strips column names. The public `CSVInspectionResult` (`columns` is a list of names, `footer_lines`, the derived `footer_rows_to_skip`) and `Usage` (attached as `result.usage`, excluded from dumps and the JSON Schema) are strict. Both share the header and dialect checks. |
| `_invokers.py` | Sync and async Ollama and Gemini invokers. They create a new client per call (thread-safe) and import SDKs lazily. Both send `response_schema()` (Ollama as `format`, retried once as `"json"` on a server that rejects a schema). Ollama's `num_ctx` is sized to the prompt. Secrets are redacted from errors, and Gemini gets one retry on 429/503, decided in one place (`_CloudCall.retry_delay`) for the sync and async calls, which differ only in their awaits. `builtin_invoker` returns an `InvokerResponse` (raw text plus token counts, retries, load time) that feeds `Usage`. |
| `_inspect.py` | `inspect_csv` / `ainspect_csv`: plan, sample, prompt, then the primary and fallback models within one time budget (`PRIMARY_SHARE` for a model followed by another), then grounding. |
| `_grounding.py` | `ground_in_samples(answer, ...)`, the only constructor of the public result in the pipeline: recomputes delimiter, header row, literal column names, header-less files and the verbatim footer (from one anchor line) from the samples, using the model's answer as the key. Each sample is parsed once (`_ParsedSample`); the header rules and the footer's anchor search and extent are separate functions. |
| `_exceptions.py` | `CSVInspectorError` and its hierarchy. |
| `cli.py`, `__main__.py` | The `csv-inspector` CLI: the only module that prints, calls `logging.basicConfig()` or reads `./.env` implicitly. |

Around the package:

- `scripts/eval_samples.py`: the manual accuracy harness (a live model
  against every `samples/manifest.json` case, per-field scores). `--out`
  streams a JSONL run (a `run` line, per-fixture verdicts, `Usage`,
  latency, a summary line; `--summarize` recovers an interrupted one),
  `--repeat` measures drift, `--subset quick|cloud` selects the documented
  lists, and quota guards (`--max-calls`, `--rpm`, `--dry-run`, ...)
  protect cloud quotas. `--replay` re-scores a `--keep-raw` run's recorded
  answers on the current code, with no model. It may import private
  names: `--keep-raw` wraps `_inspect.builtin_invoker` so token counts
  still reach `Usage`. Not run by pytest or CI.
- `scripts/compare_runs.py`: stdlib-only Markdown table of two or more
  runs, plus the fixtures whose verdict changed. The ritual, flags, quota
  notes and known limitations are in
  [`agents/csv_inspector/docs/evaluation.md`](agents/csv_inspector/docs/evaluation.md).
- `scripts/smoke_test_installed.py`: run by CI against the installed wheel
  and sdist, from outside the repository.
- `samples/`: generated, byte-exact fixtures plus `manifest.json`, written
  by `samples/generate_samples.py`. Edit the generator, never the fixtures.
  Hand-written cases live in `generate_samples.py`; combinations of quirks
  are specs in `samples/matrix.py`, rendered into `gen_*.csv` files.
- `tests/`: `conftest.py` clears the settings variables and runs each test
  in an empty directory; `fakes.py` fakes every backend; `payloads.py`
  holds the answers shared by several files. The files are split by
  concern: byte reads (`test_sampling.py`); the pipeline
  (`test_pipeline.py`); the prompt (`test_prompt.py`, `test_prompt_budget.py`);
  parsing and the answer models (`test_parsing.py`); grounding of the
  header, footer and dialect (`test_grounding_{header,footer,dialect}.py`);
  sources; time budget and async; backends;
  configuration; CLI; the embedding contract (`__all__`, no `print`, a
  `NullHandler`); the fixture catalog; the eval scoring; and the
  `docs/using-the-result.md` recipe, executed from the Markdown.
- `docs/embedding.md` (host guide) and `docs/using-the-result.md` (reader
  options) ship in the sdist.

### `examples/csv_inspector_api`

| Module | Responsibility |
|---|---|
| `app.py` | `create_app(settings=None, *, model_invoker=None, gcs_client=None)`. The two keyword arguments are the test seams. The Cloud Storage client and the inspection slots (one `asyncio.Semaphore`) are built once in the lifespan. |
| `settings.py` | `ApiSettings` (`CSV_INSPECTOR_API_*`); `to_library_settings()` is the only place that builds `csv_inspector.Settings`. |
| `routes/inspect.py` | `POST /inspect` (multipart, seekable), `/inspect/raw` (streamed, non-seekable) and `/inspect/gcs` (ranged `BlobReader`); shared query parameters, the cost guard on overrides, and the concurrency cap (`503` busy after `queue_timeout_seconds`). |
| `routes/health.py` | `GET /health`, and `?probe=true` for `ensure_backend_ready`. |
| `streaming.py` | `AsyncIteratorReader`: a blocking reader over `request.stream()` for the library's worker thread. |
| `sources/gcs.py` | Opens a `gs://` object for ranged reads; typed protocols over the untyped SDK; no `google.*` import at module level. |
| `errors.py` | One handler maps every `CSVInspectorError`, and every Cloud Storage error, to an `application/problem+json` response. |
| `request_id.py` | `RequestIdMiddleware` (the `X-Request-ID` header and the access line) and `RequestIdFilter`. |
| `main_demo.py`, `Dockerfile` | The demo (the only file that prints or configures logging) and the cloud-backend image. |

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

## Decision log

One line per notable design choice, with the issue that records it.

- **Local and free by default; cloud is an opt-in extra** with lazily
  imported SDKs: no agent needs a paid API to run.
- **Bounded sampling**: only a head and a tail window are read, never the
  whole source, whatever its size or type (paths, buffers, streams).
- **The model answers; grounding decides positions and verbatim text**:
  small models recognize structure but miscount lines, so header row,
  column names, footer and delimiter are recomputed from the samples
  ([#53](https://github.com/deluispablo/data-agent-toolkit/issues/53),
  [#97](https://github.com/deluispablo/data-agent-toolkit/issues/97)).
- **Quote escaping is read from the samples** when they show one
  convention (`\"` or a doubled quote inside a value), and quoted fields
  with neither mean no escape character; the model's answer decides only
  when they show both, or no quoted field
  ([#130](https://github.com/deluispablo/data-agent-toolkit/issues/130),
  [#158](https://github.com/deluispablo/data-agent-toolkit/issues/158)).
- **The model's delimiter yields only to a clear winner**: exactly one
  usual candidate agreeing on at least 2 head lines and at least 1.5 times
  as many as the model's answer; a wrong `,` on the tab fixtures lost by
  1.6x to 13x ([#151](https://github.com/deluispablo/data-agent-toolkit/issues/151)).
- **A footer starts at the first non-data line the model pointed at; the
  model's line is a key, matched tolerantly** (trailing empty fields
  ignored, a long enough substring accepted)
  ([#153](https://github.com/deluispablo/data-agent-toolkit/issues/153)).
- **The schema is the contract; the prompt carries semantics**: both
  backends send the same JSON Schema of the answer, so the prompt no longer
  describes the JSON shape, only what the schema cannot say (preamble vs
  header, what counts as footer, what to copy verbatim)
  ([#130](https://github.com/deluispablo/data-agent-toolkit/issues/130)).
- **What the model answers (`_ModelAnswer`) vs what the library returns
  (`CSVInspectionResult`)**: the model gives one footer anchor line,
  `footer_first_line`, and grounding reads the footer from the file;
  hosts still receive `footer_lines`. Lenient validation lives on the
  answer, strict validation on both
  ([#132](https://github.com/deluispablo/data-agent-toolkit/issues/132)).
- **One time budget per inspection**, enforced by the library even for
  custom invokers. With a fallback, the primary gets 70 % of it
  ([#12](https://github.com/deluispablo/data-agent-toolkit/issues/12),
  [#95](https://github.com/deluispablo/data-agent-toolkit/issues/95)).
- **Explicit `Settings` never read the environment**, and hosts inject
  them. `load_settings` is the opt-in loader and needs no extra
  ([#96](https://github.com/deluispablo/data-agent-toolkit/issues/96),
  [#99](https://github.com/deluispablo/data-agent-toolkit/issues/99)).
- **Configuration is checked before the source is read**, so a
  non-seekable stream is never consumed only to fail on a missing setting
  ([#82](https://github.com/deluispablo/data-agent-toolkit/issues/82)).
- **Lenient input, strict output**: common small-model spellings are
  normalized, while anything `csv` or pandas could not read fails
  validation and moves on to the fallback model
  ([#93](https://github.com/deluispablo/data-agent-toolkit/issues/93)).
- **One transient-error retry on the cloud backend**, never more; the
  fallback model handles persistent failures
  ([#98](https://github.com/deluispablo/data-agent-toolkit/issues/98)).
- **A header-less file is part of the contract** (`has_header`), not a
  sentinel value ([#94](https://github.com/deluispablo/data-agent-toolkit/issues/94)).
- **Header-less detection is a shape test on the sample, not on the
  model's examples**
  ([#131](https://github.com/deluispablo/data-agent-toolkit/issues/131)).
- **Every exception derives from `CSVInspectorError`**, and the library
  logs through module loggers only (a `NullHandler`, no `print`).

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
