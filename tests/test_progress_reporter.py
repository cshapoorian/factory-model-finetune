import pytest

from factory_model_finetune.progress_reporter import ProgressReporter


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_progress_reports_metrics():
    clock = FakeClock()
    reporter = ProgressReporter(target_tokens=1000, clock=clock)

    clock.now += 100
    reporter.accepted(250)
    reporter.rejected("duplicate")
    reporter.rejected("duplicate")
    reporter.rejected("schema")

    assert reporter.tokens_so_far == 250
    assert reporter.rejection_rates() == {"duplicate": 0.5, "schema": 0.25}
    # 250 tokens in 100 s leaves 750 tokens: 300 s.
    assert reporter.seconds_remaining() == pytest.approx(300)
    line = reporter.summary_line()
    assert "250/1,000 tokens (25.0%)" in line
    assert "duplicate 50%" in line
    assert "schema 25%" in line


def test_progress_has_no_estimate_before_any_tokens():
    reporter = ProgressReporter(target_tokens=1000, clock=FakeClock())

    assert reporter.seconds_remaining() is None
    assert "time left: unknown" in reporter.summary_line()


def test_progress_pace_ignores_tokens_from_earlier_runs():
    clock = FakeClock()
    reporter = ProgressReporter(target_tokens=1000, starting_tokens=600, clock=clock)

    clock.now += 100
    reporter.accepted(100)

    # Only 100 tokens were made this run, in 100 s; 300 tokens are left.
    assert reporter.tokens_so_far == 700
    assert reporter.seconds_remaining() == pytest.approx(300)


def test_progress_time_left_is_zero_once_target_is_reached():
    clock = FakeClock()
    reporter = ProgressReporter(target_tokens=100, clock=clock)

    clock.now += 10
    reporter.accepted(150)

    assert reporter.seconds_remaining() == 0
