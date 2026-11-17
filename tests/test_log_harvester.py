import json
import logging
from pathlib import Path

import pytest
from fakes import (
    event,
    good_job_calls,
    logged_call,
    make_spec,
    reasoning_call,
    write_job,
)

from factory_model_finetune.data_formatter import Candidate
from factory_model_finetune.log_harvester import harvest_job, job_directories
from factory_model_finetune.progress_reporter import ProgressReporter
from factory_model_finetune.schemas import SPEC_SCHEMA

APPROVED_AND_DONE = [
    event("approval", "spec", approved=True, feedback=""),
    event("approval", "plan", approved=True, feedback=""),
    event("done"),
]


@pytest.fixture
def reporter() -> ProgressReporter:
    return ProgressReporter(target_tokens=1000)


def roles_of(candidates: list[Candidate]) -> list[str]:
    return [c.role for c in candidates]


def test_harvests_valid_calls(tmp_path: Path, prompts_dir: Path, reporter: ProgressReporter):
    job = write_job(tmp_path, "sample-good", APPROVED_AND_DONE, good_job_calls(prompts_dir, 1))

    candidates = harvest_job(job, reporter)

    assert roles_of(candidates) == ["spec", "plan", "review", "commit_message"]
    spec = candidates[0]
    assert spec.source == "harvest"
    assert spec.reasoning.startswith("Let me think")
    assert "Reply with a single JSON object" in spec.system  # the production prompt, unchanged
    assert json.loads(spec.reply)["title"] == "Service 1"
    commit = candidates[3]
    assert commit.reasoning == ""
    assert commit.reply.startswith("feat(app)")


def test_skips_cancelled_jobs(tmp_path: Path, prompts_dir: Path, reporter: ProgressReporter):
    events = [event("approval", "spec", approved=True), event("cancel", "plan")]
    job = write_job(tmp_path, "sample-rejected", events, good_job_calls(prompts_dir, 1))

    assert harvest_job(job, reporter) == []
    assert reporter.rejection_rates() == {"job_outcome": 1.0}


def test_skips_rejected_jobs(tmp_path: Path, prompts_dir: Path, reporter: ProgressReporter):
    events = [
        event("approval", "spec", approved=False, feedback="You misread the request."),
        event("approval", "plan", approved=True),
        event("done"),
    ]
    job = write_job(tmp_path, "spec-sent-back", events, good_job_calls(prompts_dir, 1))

    candidates = harvest_job(job, reporter)

    assert roles_of(candidates) == ["plan", "review", "commit_message"]


def test_skips_stages_that_ended_in_an_error(
    tmp_path: Path, prompts_dir: Path, reporter: ProgressReporter
):
    events = [event("error", "plan"), event("done")]
    job = write_job(tmp_path, "plan-failed", events, good_job_calls(prompts_dir, 1))

    assert "plan" not in roles_of(harvest_job(job, reporter))


def test_skips_unapproved_calls_of_unfinished_jobs(
    tmp_path: Path, prompts_dir: Path, reporter: ProgressReporter
):
    events = [event("approval", "spec", approved=True), event("error", "implement")]
    job = write_job(tmp_path, "stalled", events, good_job_calls(prompts_dir, 1))

    assert roles_of(harvest_job(job, reporter)) == ["spec"]


def test_repair_as_final_answer(tmp_path: Path, prompts_dir: Path, reporter: ProgressReporter):
    calls = good_job_calls(prompts_dir, 1)
    spec = calls["0001-spec"]
    spec["reply"] = '{"title": "half a spec"'  # the draft the factory could not use
    repair = logged_call(
        "You convert drafts into schema-valid JSON.",
        "Convert this draft: ...",
        json.dumps(make_spec("fixed")),
        schema=SPEC_SCHEMA,
    )
    calls = {"0001-spec": spec, "0002-spec-repair": repair}
    job = write_job(tmp_path, "repaired", APPROVED_AND_DONE, calls)

    candidates = harvest_job(job, reporter)

    assert roles_of(candidates) == ["spec"]  # the repair call is not an example of its own
    assert json.loads(candidates[0].reply)["title"] == "Service fixed"
    assert candidates[0].system == spec["request"]["messages"][0]["content"]
    assert candidates[0].user == spec["request"]["messages"][1]["content"]
    assert candidates[0].reasoning == spec["reasoning"]


def test_repair_without_an_original_is_never_an_example(tmp_path: Path, reporter: ProgressReporter):
    repair = logged_call("fix it", "draft", json.dumps(make_spec(1)), schema=SPEC_SCHEMA)
    job = write_job(tmp_path, "orphan", APPROVED_AND_DONE, {"0001-spec-repair": repair})

    assert harvest_job(job, reporter) == []
    assert reporter.rejection_rates() == {"orphan_recovery": 1.0}


def test_direct_answers_are_not_paired_with_cut_off_reasoning(
    tmp_path: Path, prompts_dir: Path, reporter: ProgressReporter
):
    calls = good_job_calls(prompts_dir, 1)
    calls["0001-spec"]["reply"] = ""  # reasoning used the whole budget; no answer at all
    calls["0002-spec-direct"] = logged_call(
        "system", "user", json.dumps(make_spec(2)), schema=SPEC_SCHEMA
    )
    job = write_job(tmp_path, "direct", APPROVED_AND_DONE, {k: calls[k] for k in sorted(calls)})

    assert "spec" not in roles_of(harvest_job(job, reporter))
    assert reporter.rejection_rates()["reasoning_incomplete"] > 0


def test_skips_schema_mismatch(
    tmp_path: Path, prompts_dir: Path, reporter: ProgressReporter, caplog: pytest.LogCaptureFixture
):
    calls = good_job_calls(prompts_dir, 1)
    broken_spec = make_spec(1)
    del broken_spec["edge_cases"]
    calls["0001-spec"] = reasoning_call(
        prompts_dir, "spec", "the user prompt", broken_spec, SPEC_SCHEMA, 1
    )
    job = write_job(tmp_path, "mismatch", APPROVED_AND_DONE, calls)

    with caplog.at_level(logging.ERROR):
        candidates = harvest_job(job, reporter)

    assert roles_of(candidates) == ["plan", "review", "commit_message"]
    assert reporter.rejection_rates() == {"schema": 1.0}
    assert "does not match its schema" in caplog.text


def test_skips_unreadable_call_files(tmp_path: Path, prompts_dir: Path, reporter: ProgressReporter):
    job = write_job(tmp_path, "garbled", APPROVED_AND_DONE, good_job_calls(prompts_dir, 1))
    (job / "llm" / "0005-memory.json").write_text("{not json")

    candidates = harvest_job(job, reporter)

    assert len(candidates) == 4
    assert "log_unreadable" in reporter.rejection_rates()


def test_handles_missing_logs(
    tmp_path: Path, reporter: ProgressReporter, caplog: pytest.LogCaptureFixture
):
    with caplog.at_level(logging.WARNING):
        assert job_directories(tmp_path / "no-such-runs") == []
        assert job_directories(tmp_path) == []  # exists but holds no jobs
        no_events = tmp_path / "job-without-events"
        (no_events / "llm").mkdir(parents=True)
        assert harvest_job(no_events, reporter) == []
        empty_events = write_job(tmp_path, "job-with-empty-events", [], {})
        assert harvest_job(empty_events, reporter) == []

    assert "does not exist" in caplog.text
    assert "no jobs" in caplog.text
    assert "no events.jsonl" in caplog.text
    assert "has no events" in caplog.text


def test_job_directories_are_sorted(tmp_path: Path):
    for name in ("b-job", "a-job", "c-job"):
        (tmp_path / name).mkdir()

    assert [p.name for p in job_directories(tmp_path)] == ["a-job", "b-job", "c-job"]
