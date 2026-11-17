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
