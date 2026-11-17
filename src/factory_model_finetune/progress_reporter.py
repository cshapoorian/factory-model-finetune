"""Tracks how far the build has got, how long it will take, and which filters reject most."""

import time
from collections import Counter
from collections.abc import Callable


class ProgressReporter:
    """Tracks build progress, estimating time-to-completion and filter rejection rates.

    Args:
        target_tokens: The total number of tokens the build aims to process.
        starting_tokens: Tokens already processed before this reporter instance started.
        clock: A callable returning the current monotonic time in seconds.
    """

    def __init__(
        self,
        target_tokens: int,
        starting_tokens: int = 0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.target_tokens = target_tokens
        self.tokens_so_far = starting_tokens
        self._tokens_at_start = starting_tokens
        self._clock = clock
        self._started_at = clock()
        self._accepted = 0
        self._rejections: Counter[str] = Counter()

    def accepted(self, tokens: int) -> None:
        """Record an example that passed every filter.

        Args:
            tokens: The number of tokens generated for this accepted example.
        """
        self._accepted += 1
        self.tokens_so_far += tokens

    def rejected(self, filter_name: str) -> None:
        """Record a candidate rejected by a specific filter.

        Args:
            filter_name: The name of the filter that rejected the candidate.
        """
        self._rejections[filter_name] += 1

    def rejection_rates(self) -> dict[str, float]:
        """Calculate the rejection rate for each filter.

        The rate is defined as the number of rejections by a specific filter
        divided by the total number of candidates (accepted + all rejections).
        This provides a consistent denominator across all filters.

        Returns:
            A dictionary mapping filter names to their rejection rate (0.0 to 1.0).
            Filters with zero rejections are omitted.
        """
        total = self._accepted + sum(self._rejections.values())
        return {name: count / total for name, count in sorted(self._rejections.items())}

    def seconds_remaining(self) -> float | None:
        """Estimate time remaining until `target_tokens` is reached.

        Uses the average throughput (tokens per second) observed since the
        reporter was initialized. Returns None if insufficient data exists
        to calculate a meaningful rate (e.g., no tokens produced yet or no time elapsed).

        Returns:
            Estimated seconds remaining, or None if the rate cannot be determined.
        """
        produced = self.tokens_so_far - self._tokens_at_start
        elapsed = self._clock() - self._started_at
        if produced <= 0 or elapsed <= 0:
            return None
        return max(self.target_tokens - self.tokens_so_far, 0) / (produced / elapsed)

    def summary_line(self) -> str:
        """Generate a human-readable one-line summary of current progress.

        Includes tokens processed, percentage complete, estimated time remaining,
        and a breakdown of rejection rates by filter.

        Returns:
            A formatted string suitable for logging or console output.
        """
        percent = 100 * self.tokens_so_far / self.target_tokens
        remaining = self.seconds_remaining()
        eta = "unknown" if remaining is None else _format_duration(remaining)
        rates = ", ".join(f"{n} {r:.0%}" for n, r in self.rejection_rates().items()) or "none"
        return (
            f"{self.tokens_so_far:,}/{self.target_tokens:,} tokens ({percent:.1f}%) | "
            f"time left: {eta} | rejected by filter: {rates}"
        )


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as 'XhYYm'.

    Args:
        seconds: The duration in seconds (will be truncated to integer).

    Returns:
        A string like '2h30m'.
    """
    hours, rest = divmod(int(seconds), 3600)
    return f"{hours}h{rest // 60:02d}m"
