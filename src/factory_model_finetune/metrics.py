"""Metrics used across the analysis. Tested in tests/test_metrics.py."""


def relative_change(before: float, after: float) -> float:
    """Return (after - before) / before. Raises ValueError when before is 0."""
    if before == 0:
        raise ValueError("relative change is undefined when before is 0")
    return (after - before) / before
