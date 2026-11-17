"""Turns the factory's run logs into training candidates, keeping only good outputs.

A call is kept only when all of these hold:
- the job was not cancelled (a cancelled job's output is suspect as a whole);
- the call's stage was not sent back at a checkpoint and did not end in an error. This is judged
  per job and per stage, so a stage that was redone is dropped even if the redo was fine;
- the job reached `done`, or the call's stage was approved at a checkpoint;
- the reply parses and matches the schema the request used.

Recovery calls (`<role>-repair`, `<role>-direct`) are never examples of their own. A `-repair`
reply may stand in as the final answer to the original prompt when the first reply was invalid.
A `-direct` call answers a reasoning run that never produced JSON, so its reasoning is cut off
and there is no complete reasoning to pair with it; those are skipped.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from factory_model_finetune.data_formatter import Candidate
from factory_model_finetune.progress_reporter import ProgressReporter
from factory_model_finetune.roles import ALL_ROLES, REASONING_ROLES
from factory_model_finetune.schemas import (
    ROLE_SCHEMAS,
    SCHEMA_FILTER,
    Schema,
    schema_in_system_prompt,
    valid_reply,
)

logger = logging.getLogger(__name__)

SOURCE = "harvest"
RECOVERY_SUFFIXES = ("repair", "direct")

# The pipeline stage each role runs in; `summarize` runs in several, so it follows the job outcome.
STAGE_OF_ROLE = {
    "detect_profile": "setup",
    "spec": "spec",
    "plan": "plan",
    "trivial_edit": "implement",
    "review": "review",
    "docs_pass": "review",
    "evidence": "preview",
    "commit_message": "pr",
    "pr_body": "pr",
    "memory": "pr",
}

# Names under which harvest rejections are reported.
JOB_OUTCOME = "job_outcome"
LOG_UNREADABLE = "log_unreadable"
ORPHAN_RECOVERY = "orphan_recovery"
REASONING_INCOMPLETE = "reasoning_incomplete"


@dataclass(frozen=True)
class JobOutcome:
    """What the job's event log says about how it went."""

    reached_done: bool
    was_cancelled: bool
    approved_stages: frozenset[str]
    unusable_stages: frozenset[str]  # sent back at a checkpoint, or ended in an error

    def allows(self, role: str) -> bool:
        """True if outputs of `role` may be trained on."""
        stage = STAGE_OF_ROLE.get(role, "")
        if stage in self.unusable_stages:
            return False
        return self.reached_done or stage in self.approved_stages


@dataclass(frozen=True)
class LoggedCall:
    """One `llm/NNNN-<role>.json` file."""

    base_role: str
    recovery: str  # "", "repair" or "direct"
    request: dict[str, Any]
    reasoning: str
    reply: str


@dataclass
class _Attempt:
    original: LoggedCall
    recovery: LoggedCall | None = None


def job_directories(runs_dir: Path) -> list[Path]:
    """Return the job folders under `runs_dir` in name order; warn and return none if absent.

    Args:
        runs_dir: Path to the directory containing job subdirectories.

    Returns:
        Sorted list of job directory paths. Empty list if the directory is missing or empty.
    """
    if not runs_dir.is_dir():
        logger.warning("factory runs folder %s does not exist; harvesting nothing", runs_dir)
        return []
    jobs = sorted(path for path in runs_dir.iterdir() if path.is_dir())
    if not jobs:
        logger.warning("factory runs folder %s has no jobs; harvesting nothing", runs_dir)
    return jobs


def harvest_job(job_dir: Path, reporter: ProgressReporter) -> list[Candidate]:
    """Extract valid training candidates from a single job's logs.

    Filters out cancelled jobs, unreadable logs, and calls that fail schema validation
    or lack required reasoning. Rejections are reported via `reporter`.

    Args:
        job_dir: Path to the job directory containing `events.jsonl` and `llm/` subfolder.
        reporter: ProgressReporter instance to log rejections.

    Returns:
        List of valid `Candidate` objects ready for training.
    """
    outcome = read_outcome(job_dir / "events.jsonl")
    if outcome is None:
        return []
    if outcome.was_cancelled:
        logger.info("job %s was cancelled; none of its calls are used", job_dir.name)
        reporter.rejected(JOB_OUTCOME)
        return []
    candidates = []
    for attempt in _pair_recoveries(read_calls(job_dir / "llm", reporter), reporter):
        if not outcome.allows(attempt.original.base_role):
            reporter.rejected(JOB_OUTCOME)
            continue
        candidate = _candidate_from(attempt, job_dir.name, reporter)
        if candidate:
            candidates.append(candidate)
    return candidates


def read_outcome(events_path: Path) -> JobOutcome | None:
    """Parse `events.jsonl` to determine job status and stage validity.

    Reads the event log to identify if the job completed, was cancelled, or had stages
    marked as approved or unusable (error/redone). Returns None if the file is missing
    or contains no valid events.

    Args:
        events_path: Path to the `events.jsonl` file.

    Returns:
        A `JobOutcome` summarizing the job's state, or None if unreadable/empty.
    """
    if not events_path.is_file():
        logger.warning("no events.jsonl at %s; skipping the job (no labels)", events_path)
        return None
    reached_done = was_cancelled = False
    approved: set[str] = set()
    unusable: set[str] = set()
    event_count = 0
    for line in events_path.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            logger.warning("skipping unreadable line in %s: %s", events_path, error)
            continue
        event_count += 1
        kind, stage = event.get("kind"), event.get("stage") or ""
        if kind == "done":
            reached_done = True
        elif kind == "cancel":
            was_cancelled = True
        elif kind == "error":
            unusable.add(stage)
        elif kind == "approval":
            (approved if (event.get("data") or {}).get("approved") else unusable).add(stage)
    if event_count == 0:
        logger.warning("%s has no events; skipping the job (no labels)", events_path)
        return None
    return JobOutcome(reached_done, was_cancelled, frozenset(approved), frozenset(unusable))


def read_calls(llm_dir: Path, reporter: ProgressReporter) -> list[LoggedCall]:
    """Load all LLM call logs from a job's `llm/` directory.

    Files are expected to be named `NNNN-<role>.json` (or with recovery suffixes).
    Unreadable files are skipped and reported as rejections.

    Args:
        llm_dir: Path to the directory containing call log JSON files.
        reporter: ProgressReporter instance to log unreadable file rejections.

    Returns:
        List of `LoggedCall` objects parsed from the JSON files, sorted by filename.
    """
    calls = []
    for path in sorted(llm_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            calls.append(_logged_call(path.stem.partition("-")[2], data))
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as error:
            logger.error("skipping unreadable call log %s: %r", path, error)
            reporter.rejected(LOG_UNREADABLE)
    return calls


def _logged_call(role: str, data: dict[str, Any]) -> LoggedCall:
    base_role, _, recovery = role.rpartition("-")
    if recovery not in RECOVERY_SUFFIXES:
        base_role, recovery = role, ""
    return LoggedCall(
        base_role=base_role,
        recovery=recovery,
        request=data["request"],
        reasoning=data.get("reasoning") or "",
        reply=data["reply"],
    )


def _pair_recoveries(calls: list[LoggedCall], reporter: ProgressReporter) -> list[_Attempt]:
    """Pair recovery calls (repair/direct) with their corresponding original calls.

    Iterates through calls in order. When an original call is encountered, it is stored
    as the latest for its role. When a recovery call is encountered, it is attached to
    the most recent original call for that role. Recovery calls without a preceding
    original call (orphans) are dropped and reported.

    Args:
        calls: List of `LoggedCall` objects in chronological order.
        reporter: ProgressReporter instance to log orphan recovery rejections.

    Returns:
        List of `_Attempt` objects, each containing an original call and an optional recovery.
    """
    attempts: list[_Attempt] = []
    latest: dict[str, _Attempt] = {}
    for call in calls:
        if not call.recovery:
            latest[call.base_role] = _Attempt(call)
            attempts.append(latest[call.base_role])
            continue
        attempt = latest.get(call.base_role)
        if attempt is None or attempt.recovery is not None:
            logger.error("recovery call for %s has no original call; ignoring it", call.base_role)
            reporter.rejected(ORPHAN_RECOVERY)
            continue
        attempt.recovery = call
    return attempts


def _candidate_from(
    attempt: _Attempt, job_name: str, reporter: ProgressReporter
) -> Candidate | None:
    """Construct a `Candidate` from a paired original and optional recovery call.

    Validates that the role is known, extracts system/user prompts, checks reasoning
    completeness for reasoning roles, and verifies the reply against the expected schema.
    If the original reply fails schema validation, the recovery reply (if available and
    of type 'repair') is tried as a fallback.

    Args:
        attempt: The paired original and optional recovery call.
        job_name: Name of the job for logging context.
        reporter: ProgressReporter instance to log validation failures.

    Returns:
        A `Candidate` object if valid, or None if validation fails.
    """
    call = attempt.original
    role = call.base_role
    if role not in ALL_ROLES:
        logger.debug("job %s: no training use for role %r", job_name, role)
        return None
    prompts = _system_and_user(call.request)
    if prompts is None:
        logger.error("job %s: %s call has no system and user messages", job_name, role)
        reporter.rejected(LOG_UNREADABLE)
        return None
    system, user = prompts
    reasoning = call.reasoning.strip()
    recovery = attempt.recovery
    # Reasoning roles require reasoning text. Direct recoveries are skipped because
    # they answer a reasoning run that never produced JSON, so there is no complete
    # reasoning to pair with the direct reply.
    if role in REASONING_ROLES and (not reasoning or (recovery and recovery.recovery == "direct")):
        reporter.rejected(REASONING_INCOMPLETE)
        return None
    schema = _schema_of(call.request, role, system)
    answer = valid_reply(call.reply, schema)
    # If the original reply is invalid, try the repair reply as a fallback.
    if answer is None and recovery and recovery.recovery == "repair":
        answer = valid_reply(recovery.reply, schema)
    if answer is None:
        logger.error("job %s: %s reply does not match its schema; skipping", job_name, role)
        reporter.rejected(SCHEMA_FILTER)
        return None
    return Candidate(
        role, SOURCE, system, user, reasoning if role in REASONING_ROLES else "", answer
    )


def _system_and_user(request: dict[str, Any]) -> tuple[str, str] | None:
    """Extract system and user message contents from a request.

    Expects exactly two messages: first 'system', second 'user'. Returns None if the
    structure is invalid.

    Args:
        request: The LLM request dictionary containing 'messages'.

    Returns:
        Tuple of (system_content, user_content) strings, or None if invalid.
    """
    messages = request.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        return None
    system, user = messages
    if system.get("role") != "system" or user.get("role") != "user":
        return None
    return system["content"], user["content"]


def _schema_of(request: dict[str, Any], role: str, system: str) -> Schema | None:
    """Determine the schema used to validate the reply.

    Priority order:
    1. Explicit schema in the request's `response_format`.
    2. Schema embedded in the system prompt.
    3. Default schema for the role from `ROLE_SCHEMAS`.

    Args:
        request: The LLM request dictionary.
        role: The role of the call.
        system: The system prompt content.

    Returns:
        The schema object to use for validation, or None if no schema is found.
    """
    format_schema = (request.get("response_format") or {}).get("json_schema", {}).get("schema")
    return format_schema or schema_in_system_prompt(system) or ROLE_SCHEMAS.get(role)
