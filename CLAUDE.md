# factory-model-finetune

Built with the `data-science-analysis` profile.

# Conventions (all projects)

## Readability first
- Code is read far more than it's written. Optimize for the next person (or model) reading it cold.
- Small, single-purpose functions with names that say what they do. A function that needs "and" in its name is two functions.
- Names are specific and honest: `unpaid_invoices`, not `data`; `parse_iso_date`, not `handle`. Booleans read as questions (`is_active`, `has_access`).
- Prefer plain functions and data over classes and inheritance. Add an abstraction only when there are already two real uses for it.
- Keep nesting shallow: return early, guard clauses over nested `if`s.
- No dead code, no commented-out code, no speculative options or parameters "for later".

## Simplicity
- Use the standard library and what the project already has before adding a dependency. Every new dependency needs a reason in the spec.
- The smallest change that fully meets the spec wins. Don't refactor unrelated code in the same change.

## Errors and edge cases
- Validate input at the boundary (CLI args, HTTP requests, files, env) and fail with a message that names the bad value and what was expected.
- Never swallow errors. Catch only what you can handle; let the rest propagate.
- Handle every edge case listed in the spec explicitly, and test it.

## Comments
- Comments explain why, not what: intent, constraints, units, invariants, surprising choices.
- Public functions, classes and modules get a short docstring: what it does, important parameters, return value, how it fails.

## Tests
- Test logic and behaviour, not implementation details. One behaviour per test, named after it (`test_rejects_negative_amount`).
- Cover the acceptance criteria and edge cases from the spec. Don't test UI layout or styling.
- Tests are deterministic: no real network, fixed seeds, fixed clocks, temp dirs.

## Safety
- No secrets in code, tests or logs. Read configuration from the environment and document every variable in the README.
- Never log personal data or credentials.

# Python conventions
- Python 3.12+, managed with uv. Add dependencies with `uv add` (dev tools: `uv add --dev`); never pip.
- src layout: code in `src/<package>/`, tests in `tests/` mirroring the module structure.
- Type hints on every function signature; pyright must pass. Prefer built-in generics (`list[str]`, `dict[str, int]`, `X | None`).
- Use `dataclasses` (or Pydantic at I/O boundaries) for structured data, not loose dicts passed around.
- `pathlib.Path` for paths, `logging` (not print) for diagnostics in libraries; CLIs print results to stdout and errors to stderr with a non-zero exit code.
- Raise specific exceptions (`ValueError`, custom ones subclassing a package base error) with messages naming the bad value.
- Docstrings: one-line summary, then Args/Returns/Raises only where not obvious.
- ruff formats and lints; don't fight it, and don't add `noqa` without a comment saying why.
- pytest: plain `assert`, fixtures for setup, `tmp_path` for files, `pytest.mark.parametrize` for input tables.

# Data science conventions
- Logic lives in `src/<package>/` modules (loading, features, models, metrics) and is unit-tested. Notebooks and `reports/` only call those modules and present results.
- Reproducible: fixed random seeds passed explicitly, data snapshots identified by path and hash, dependencies locked.
- Start with exploratory analysis: shape, types, missingness, distributions, leakage checks. Write down what you found before modelling.
- Statistics: state the hypothesis, check test assumptions, report effect sizes with confidence intervals, not only p-values. Correct for multiple comparisons.
- Modelling: split before any fitting (no leakage), use pipelines for preprocessing, cross-validate, compare against a simple baseline, and report the metric that matches the question.
- Every number in a report comes from code that's tested; charts have titles, labeled axes with units, and a sentence saying what they show.
- Reports end with findings, caveats and limitations in plain language.
