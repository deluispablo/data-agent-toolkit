## What and why

<!-- What does this change, and why? Link the issue it resolves: "Closes #123". -->

## How it was verified

<!-- Tests added or updated, commands run, manual checks (e.g. eval_samples.py against a live model). -->

## Checklist

- [ ] `uv run ruff check .` and `uv run ruff format --check .` pass.
- [ ] `uv run mypy` and `uv run pytest --cov` pass in every agent touched.
- [ ] New behaviour is covered by tests that need no network, Ollama or credentials.
- [ ] User-visible changes are listed under `[Unreleased]` in the agent's `CHANGELOG.md`
      (breaking ones under **Changed (breaking)**).
- [ ] Public API changes are documented (docstrings, the agent's README and `docs/`).
