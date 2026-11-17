"""Quality filters for examples that already match their schema.

An example must be substantial (`non_trivial`), agree with its own prompt (`consistency`: for
instance every AC and EC is covered by some task) and not repeat an earlier one (`duplicate`).
Schema validity is checked where the example comes from, because a log's schema can differ from
the built-in one.
"""

import hashlib
import json
import logging
import re
from collections.abc import Callable
from typing import Any

from factory_model_finetune.data_formatter import Candidate
from factory_model_finetune.factory_text import is_comment_only_change, replace_once
from factory_model_finetune.roles import REASONING_ROLES
from factory_model_finetune.schemas import ROLE_SCHEMAS

logger = logging.getLogger(__name__)

NON_TRIVIAL = "non_trivial"
CONSISTENCY = "consistency"
DUPLICATE = "duplicate"

MIN_REASONING_CHARS = 200
MIN_TEXT_CHARS = {"summarize": 40, "commit_message": 40, "pr_body": 150}
MIN_DETAILS_CHARS = 30
DUPLICATE_OVERLAP = 0.9
MIN_WORDS_TO_COMPARE = 12
BLOCKING_SEVERITIES = ("blocker", "major")
SPEC_ID_IN_MARKDOWN = re.compile(r"\*\*((?:AC|EC)\d+)\*\*")
SPEC_ID_IN_JSON = re.compile(r'"id":\s*"((?:AC|EC)\d+)"')
CONVENTIONAL_SUBJECT = re.compile(r"^[a-z]+(\([^)]+\))?!?: \S")


class DuplicateIndex:
    """Remembers earlier examples and spots near-copies by word overlap.

    Replies are only compared with earlier replies to the same system prompt (so, the same
    role), and only with those of a similar word count, so a lookup stays cheap however many
    examples are stored.
    """

    def __init__(self, threshold: float = DUPLICATE_OVERLAP) -> None:
        self._threshold = threshold
        self._prompt_digests: set[bytes] = set()
        # system prompt digest -> word count -> word sets of the replies with that count
        self._replies: dict[bytes, dict[int, list[frozenset[str]]]] = {}

    def is_duplicate(self, system: str, user: str, reply: str) -> bool:
        """True if this prompt was seen before, or `reply` shares over 90% of its words with one.

        The reply is compared, not the prompt, because prompts of one profile share long
        boilerplate (the conventions) and would all look alike. Replies under
        `MIN_WORDS_TO_COMPARE` words are skipped: short answers such as a profile pick are
        legitimately alike.
        """
        if _digest(system, user) in self._prompt_digests:
            return True
        words = frozenset(reply.split())
        if len(words) < MIN_WORDS_TO_COMPARE:
            return False
        by_count = self._replies.get(_digest(system), {})
        # A set can only overlap by more than the threshold with one of a similar size.
        for count in range(
            int(len(words) * self._threshold), int(len(words) / self._threshold) + 1
        ):
            if any(
                _overlap(words, earlier) > self._threshold for earlier in by_count.get(count, [])
            ):
                return True
        return False

    def add(self, system: str, user: str, reply: str) -> None:
        """Remember an accepted example."""
        self._prompt_digests.add(_digest(system, user))
        words = frozenset(reply.split())
        self._replies.setdefault(_digest(system), {}).setdefault(len(words), []).append(words)


def _digest(*parts: str) -> bytes:
    """Hash text with SHA-256; unlike `hash()` it does not change from one process to the next."""
    return hashlib.sha256("\x00".join(parts).encode()).digest()


def _overlap(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard similarity of two word sets."""
    return len(a & b) / len(a | b)


def failed_filter(candidate: Candidate, index: DuplicateIndex) -> str | None:
    """Return the name of the first filter `candidate` fails, or None if it passes them all.

    The candidate's reply must already be valid for its role (JSON for the JSON roles).
    """
    value = json.loads(candidate.reply) if candidate.role in ROLE_SCHEMAS else candidate.reply
    checks = (
        (NON_TRIVIAL, _triviality_problem(candidate, value)),
        (CONSISTENCY, _inconsistency(candidate.role, value, candidate.user)),
    )
    for name, problem in checks:
        if problem:
            logger.debug("%s example failed %s: %s", candidate.role, name, problem)
            return name
    if index.is_duplicate(candidate.system, candidate.user, candidate.reply):
        return DUPLICATE
    return None


# ---- non-triviality ----------------------------------------------------------


def _triviality_problem(candidate: Candidate, value: Any) -> str | None:
    if candidate.role in REASONING_ROLES and len(candidate.reasoning.strip()) < MIN_REASONING_CHARS:
        return "reasoning is too short"
    if candidate.role in MIN_TEXT_CHARS and len(value.strip()) < MIN_TEXT_CHARS[candidate.role]:
        return "text reply is too short"
    check = _TRIVIALITY_CHECKS.get(candidate.role)
    return check(value) if check else None


def _spec_is_thin(spec: dict[str, Any]) -> str | None:
    """Ensure the spec has enough substance to generate meaningful work.

    Rejects specs that are too thin to define a real task: fewer than 2 acceptance criteria,
    no edge cases, or missing architectural components. Also enforces that the goal and summary
    are substantive sentences (>= 40 chars) to prevent vague instructions.
    """
    if len(spec["acceptance_criteria"]) < 2 or not spec["edge_cases"]:
        return "needs at least 2 acceptance criteria and 1 edge case"
    if len(spec.get("goal", spec["summary"])) < 40 or len(spec["summary"]) < 40:
        return "goal and summary must each be a real sentence"
    return None if spec["architecture"]["components"] else "no components"


def _plan_is_thin(plan: dict[str, Any]) -> str | None:
    """Ensure the plan has actionable tasks with sufficient detail.

    Rejects plans with no tasks or tasks that lack substantive implementation details
    (fewer than `MIN_DETAILS_CHARS` characters). This prevents trivial or placeholder plans.
    """
    if not plan["tasks"]:
        return "no tasks"
    if any(len(task["details"]) < MIN_DETAILS_CHARS for task in plan["tasks"]):
        return "a task has no real details"
    return None


def _review_is_thin(review: dict[str, Any]) -> str | None:
    """Ensure the review summary is substantive.

    A summary shorter than 20 characters is considered too brief to provide useful feedback.
    """
    return None if len(review["summary"]) >= 20 else "summary is too short"


def _mapping_is_thin(mapping: dict[str, Any]) -> str | None:
    """Ensure the evidence mapping covers at least some criteria.

    Rejects mappings that have no criteria mapped, indicating a failure to connect
    test results to the specification.
    """
    return None if mapping["criteria"] else "no criteria mapped"


def _edits_are_thin(reply: dict[str, Any]) -> str | None:
    """Ensure the edit reply contains actual changes.

    Rejects replies with an empty edits list, which indicates no work was performed.
    """
    return None if reply["edits"] else "no edits"


def _memory_is_thin(lessons: dict[str, Any]) -> str | None:
    """Ensure the memory entry contains at least one lesson.

    Rejects entries where both 'you' (personal insights) and 'decisions' (rationales)
    are empty, as this provides no value for future context.
    """
    return None if lessons["you"] or lessons["decisions"] else "no lessons"


def _profile_pick_is_thin(pick: dict[str, Any]) -> str | None:
    """Ensure the profile selection includes a justification.

    Rejects picks with a 'why' explanation shorter than 10 characters, ensuring
    the model provides some reasoning for its choice.
    """
    return None if len(pick["why"]) >= 10 else "no reason given"


_TRIVIALITY_CHECKS: dict[str, Callable[[Any], str | None]] = {
    "spec": _spec_is_thin,
    "plan": _plan_is_thin,
    "review": _review_is_thin,
    "evidence": _mapping_is_thin,
    "trivial_edit": _edits_are_thin,
    "docs_pass": _edits_are_thin,
    "memory": _memory_is_thin,
    "detect_profile": _profile_pick_is_thin,
}


# ---- consistency -------------------------------------------------------------


def _inconsistency(role: str, value: Any, prompt: str) -> str | None:
    if isinstance(value, str) and re.search(r"</?think>", value):
        return "reasoning leaked into the reply"
    check = _CONSISTENCY_CHECKS.get(role)
    return check(value, prompt) if check else None


def _spec_inconsistency(spec: dict[str, Any], _prompt: str) -> str | None:
    """Validate spec structure: unique IDs and correct format.

    Checks that acceptance criteria and edge cases have unique IDs matching the
    expected format (e.g., 'AC1', 'EC2'). Duplicate or malformed IDs indicate
    a generation error.
    """
    for key, prefix in (("acceptance_criteria", "AC"), ("edge_cases", "EC")):
        ids = [item["id"] for item in spec[key]]
        if len(set(ids)) != len(ids):
            return f"duplicate ids in {key}"
        if not all(re.fullmatch(rf"{prefix}\d+", i) for i in ids):
            return f"ids in {key} must look like {prefix}1"
    return None


def _plan_inconsistency(plan: dict[str, Any], prompt: str) -> str | None:
    """Validate plan against the prompt's specification requirements.

    Ensures every AC/EC in the prompt is covered by exactly one task, no task
    covers non-existent IDs, and trivial tasks specify files. This enforces
    completeness and correctness of the plan relative to the spec.
    """
    spec_ids = set(SPEC_ID_IN_MARKDOWN.findall(prompt))
    if not spec_ids:
        return "the prompt lists no AC or EC ids to cover"
    task_ids = [task["id"] for task in plan["tasks"]]
    if len(set(task_ids)) != len(task_ids):
        return "duplicate task ids"
    covered = {c.strip() for task in plan["tasks"] for c in task["covers"]}
    if unknown := covered - spec_ids:
        return f"covers ids that are not in the spec: {sorted(unknown)}"
    if missing := spec_ids - covered:
        return f"no task covers {sorted(missing)}"
    if any(task["kind"] == "trivial" and not task["files"] for task in plan["tasks"]):
        return "a trivial task lists no files"
    return None


def _review_inconsistency(review: dict[str, Any], prompt: str) -> str | None:
    """Validate review verdict consistency and finding references.

    Checks that 'pass' verdicts have no blocking/major findings, 'changes' verdicts
    have findings, and all referenced files exist in the prompt's diff. Ensures
    logical consistency between the verdict and the detailed findings.
    """
    findings = review["findings"]
    has_blocking = any(f["severity"] in BLOCKING_SEVERITIES for f in findings)
    if review["verdict"] == "pass" and has_blocking:
        return "verdict is pass but a blocker or major finding is listed"
    if review["verdict"] == "changes" and not findings:
        return "verdict is changes but there are no findings"
    for finding in findings:
        if finding["line"] < 0:
            return "a finding has a negative line"
        if finding["file"] and finding["file"] not in prompt:
            return f"finding names {finding['file']!r}, which is not in the diff"
    return None


def _evidence_inconsistency(mapping: dict[str, Any], prompt: str) -> str | None:
    """Validate evidence mapping against prompt requirements and test results.

    Ensures all AC/EC IDs from the prompt are mapped, and that all tests referenced
    in the mapping were actually listed as run in the prompt. This prevents hallucinated
    test results or missing coverage.
    """
    prompt_ids = SPEC_ID_IN_JSON.findall(prompt)
    for key, prefix in (("criteria", "AC"), ("edge_cases", "EC")):
        expected = sorted(i for i in prompt_ids if i.startswith(prefix))
        mapped = sorted(entry["id"] for entry in mapping[key])
        if mapped != expected:
            return f"{key} ids {mapped} do not match the prompt's {expected}"
    listed = _listed_tests(prompt)
    if listed is None:
        return "cannot read the test list from the prompt"
    named = {
        name for key in ("criteria", "edge_cases") for e in mapping[key] for name in e["tests"]
    }
    if missing := named - listed:
        return f"tests that were not run: {sorted(missing)}"
    return None


def _listed_tests(prompt: str) -> set[str] | None:
    _, _, tail = prompt.partition("# Tests that ran\n")
    try:
        return set(json.loads(tail))
    except (json.JSONDecodeError, TypeError):
        return None


def _section(prompt: str, start_marker: str, end_marker: str) -> str:
    _, _, rest = prompt.partition(start_marker)
    return rest.rpartition(end_marker)[0] if end_marker in rest else rest


def _trivial_edit_inconsistency(reply: dict[str, Any], prompt: str) -> str | None:
    """Validate trivial edits against the current file state.

    Checks that edits either append to new files or find/replace in existing files
    with exact matches. Rejects empty finds on existing files or find texts that
    don't match exactly once in the target file content.
    """
    try:
        files: dict[str, str] = json.loads(
            _section(prompt, "# Current files\n", "\n\n# Conventions")
        )
    except json.JSONDecodeError:
        return "cannot read the files from the prompt"
    for edit in reply["edits"]:
        path = edit["path"].lstrip("/")
        if not edit["find"]:
            if path in files:
                return f"{path}: empty find on an existing file"
            files[path] = edit["replace"]
            continue
        updated = replace_once(files.get(path, ""), edit["find"], edit["replace"])
        if updated is None:
            return f"{path}: find text must match exactly once in an existing file"
        files[path] = updated
    return None


def _docs_inconsistency(reply: dict[str, Any], prompt: str) -> str | None:
    """Validate docs edits are comment-only changes.

    Applies the edits to the file content and checks if the resulting change
    is purely cosmetic (comments/docstrings). Rejects edits that modify code logic,
    as this role is for documentation updates only.
    """
    first_line, _, rest = prompt.partition("\n\n")
    path = first_line.removeprefix("# File: ")
    before = rest.rpartition("\n\n# Conventions")[0]
    after = before
    for edit in reply["edits"]:
        updated = replace_once(after, edit["find"], edit["replace"])
        if updated is None:
            return "find text must match exactly once in the file"
        after = updated
    return (
        None
        if is_comment_only_change(path, before, after)
        else "edits change code, not only comments"
    )


def _memory_inconsistency(lessons: dict[str, Any], prompt: str) -> str | None:
    """Ensure lessons are new and not duplicates of existing notes.

    Checks that no lesson (personal insight or decision rationale) already exists
    in the current notes. This prevents redundant memory entries.
    """
    notes = _section(prompt, "# Current notes\n", "\n\n# The job").lower()
    for lesson in lessons["you"] + lessons["decisions"]:
        if lesson.strip().lower() in notes:
            return f"lesson is already in the notes: {lesson!r}"
    return None


def _profile_pick_inconsistency(pick: dict[str, Any], prompt: str) -> str | None:
    """Ensure the selected profile was actually offered in the prompt.

    Rejects picks where the chosen profile name is not found in the prompt,
    indicating the model hallucinated an option.
    """
    return None if f'"{pick["profile"]}"' in prompt else "picked a profile that was not offered"


def _commit_inconsistency(message: str, _prompt: str) -> str | None:
    """Validate commit message format and conventions.

    Checks for Conventional Commits style (type(scope): subject), length limits
    (< 72 chars), blank second line, and absence of code fences. Ensures the
    message adheres to standard git commit formatting rules.
    """
    lines = message.strip().splitlines()
    if "```" in message:
        return "commit message contains a code fence"
    if not CONVENTIONAL_SUBJECT.match(lines[0]) or len(lines[0]) >= 72:
        return "subject must be Conventional Commits style and under 72 characters"
    if len(lines) > 1 and lines[1]:
        return "second line must be blank"
    return None


_CONSISTENCY_CHECKS: dict[str, Callable[[Any, str], str | None]] = {
    "spec": _spec_inconsistency,
    "plan": _plan_inconsistency,
    "review": _review_inconsistency,
    "evidence": _evidence_inconsistency,
    "trivial_edit": _trivial_edit_inconsistency,
    "docs_pass": _docs_inconsistency,
    "memory": _memory_inconsistency,
    "detect_profile": _profile_pick_inconsistency,
    "commit_message": _commit_inconsistency,
}
