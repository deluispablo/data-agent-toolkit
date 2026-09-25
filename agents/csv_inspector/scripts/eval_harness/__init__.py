"""The manual evaluation harness of csv_inspector: live runs, replays and their reports.

Run it through ``scripts/eval_samples.py`` (the harness) and
``scripts/compare_runs.py`` (the multi-run table); ``docs/evaluation.md``
describes the ritual. Not run by pytest or CI: it calls a real model.

Modules, from the ground up: :mod:`.scoring` (a result against the manifest,
repeat votes), :mod:`.replay` (recorded answers), :mod:`.evaluation` (one
inspection, as one run-file line), :mod:`.guards` (fixture selection and
quota guards), :mod:`.runs` (run files, the guarded loop, summaries),
:mod:`.report` (standard library only: every human and Markdown rendering)
and :mod:`.cli`. Nothing is imported here, so ``compare_runs.py`` runs
without the package installed.
"""
