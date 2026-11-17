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
