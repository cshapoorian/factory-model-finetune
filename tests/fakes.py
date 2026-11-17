"""Test doubles and fixtures: a fake LM Studio client and logs shaped like factory-log-samples."""

import json
import re
from collections.abc import Collection
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer, models, pre_tokenizers

from factory_model_finetune.errors import LmStudioError
from factory_model_finetune.factory_text import (
    guided_system,
    plan_markdown,
    render_prompt,
    spec_markdown,
)
from factory_model_finetune.lm_client import Completion
from factory_model_finetune.schemas import PLAN_SCHEMA, REVIEW_SCHEMA, SPEC_SCHEMA, Schema

# The factory's real user templates (placeholders and headings intact); system prompts are short
# stand-ins because only the user templates' structure is parsed by the filters.
USER_TEMPLATES = {
    "spec_user": "# Request\n$request\n\n# Profile\n$profile\n\n"
    "# Conventions this project follows\n$conventions\n\n# Repository\n$repo\n\n"
    "# What we know about the person you're building for\n$memory\n\n"
    "# Previous spec (if this is a revision)\n$previous\n\n"
    "# Feedback on the previous spec (address every point)\n$feedback\n\n"
    "# Reference material attached to this job (read-only; the request may refer to it)\n"
    "$references\n",
    "plan_user": "# Spec\n$spec\n\n# Repository\n$repo\n\n# Files most relevant to this request\n"
    "$relevant\n\n# Conventions\n$conventions\n\n# Notes from the person at earlier checkpoints\n"
    "$notes\n\n# Feedback on the previous plan (address every point)\n$feedback\n\n"
    "# Reference material attached to this job\n$references\n",
    "review_user": "# Spec\n$spec\n\n# Conventions\n$conventions\n\n# Review rubric\n$rubric\n\n"
    "# Diff\n```diff\n$diff\n```\n",
    "evidence_user": "# Acceptance criteria\n$criteria\n\n# Edge cases\n$edge_cases\n\n"
    "# Tests that ran\n$tests\n",
    "trivial_edit_user": "# Task\n$task\n\n# Current files\n$files\n\n"
    "# Conventions\n$conventions\n",
    "docs_user": "# File: $path\n\n$content\n\n# Conventions\n$conventions\n",
    "pr_user": "# Spec\n$spec\n\n# Evidence\n$evidence\n\n# Open review findings\n$open_findings\n",
    "profile_user": "# Request\n$request\n\n# Profiles\n$profiles\n\n# Repository\n$repo\n",
    "memory_user": "# Current notes\n$memory\n\n# The job\n$request\n\n# Their feedback during the "
    "job\n$feedback\n",
    "answer_user": "# Context\n$context\n\n# Their message\n$question\n",
}
SYSTEM_NAMES = (
    "spec",
    "plan",
    "review",
    "evidence",
    "trivial_edit",
    "docs",
    "summarize",
    "commit",
    "pr",
    "profile",
    "memory",
    "answer",
)
EXAMPLE_PATH = "src/app/settings.py"
SERVICE_PATH = "src/app/service.py"


def write_prompts(prompts_dir: Path) -> Path:
    """Write the prompt templates the generator renders and return the folder."""
    prompts_dir.mkdir(parents=True, exist_ok=True)
    for name, text in USER_TEMPLATES.items():
        (prompts_dir / f"{name}.md").write_text(text)
    for name in SYSTEM_NAMES:
        (prompts_dir / f"{name}_system.md").write_text(f"You are the {name} role. Reply carefully.")
    return prompts_dir


def write_tokenizer(path: Path) -> Path:
    """Write a tiny tokenizer that counts one token per word or punctuation run."""
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()  # type: ignore[assignment]
    tokenizer.save(str(path))
    return path


def words(tag: object, count: int = 12) -> str:
    """Return `count` words that are unique to `tag`, so no two replies look like duplicates."""
    return " ".join(f"w{tag}x{i}" for i in range(count))


def make_spec(tag: object, criteria: int = 3, edge_cases: int = 2) -> dict[str, Any]:
    return {
        "goal": f"The person wants a service that does thing {tag} for its users. {words(tag)}",
        "title": f"Service {tag}",
        "summary": f"A small service that does thing {tag} and nothing else. {words(tag)}",
        "acceptance_criteria": [
            {"id": f"AC{i}", "text": f"Given input {i}, when run, then output {tag} {i}."}
            for i in range(1, criteria + 1)
        ],
        "edge_cases": [
            {"id": f"EC{i}", "text": f"Given bad input {i}, then fail clearly {tag}."}
            for i in range(1, edge_cases + 1)
        ],
        "out_of_scope": [f"Extras {tag}"],
        "assumptions": [f"Python is installed {tag}"],
        "architecture": {
            "overview": f"One module {tag}. {words(tag)}",
            "components": [{"name": f"Core{tag}", "responsibility": f"Does the work {tag}"}],
            "data_model": f"Record {tag}",
            "decisions": [{"decision": f"Use stdlib {tag}", "why": f"Simplest {tag}"}],
        },
    }


def make_plan(
    tag: object, covers: Collection[str], files: list[str] | None = None
) -> dict[str, Any]:
    return {
        "tasks": [
            {
                "id": "T1",
                "title": f"Build it {tag}",
                "details": f"Write the module and its tests in a clear way. {words(tag)}",
                "files": files or [SERVICE_PATH],
                "kind": "code",
                "tests": [f"test_behaviour_{tag}_a", f"test_behaviour_{tag}_b"],
                "covers": sorted(covers),
            }
        ]
    }


def make_review(tag: object, *, passes: bool) -> dict[str, Any]:
    findings = (
        []
        if passes
        else [
            {
                "severity": "major",
                "file": SERVICE_PATH,
                "line": 3,
                "issue": f"Error is swallowed {tag}",
                "suggestion": f"Let it propagate {tag}",
            }
        ]
    )
    return {
        "verdict": "pass" if passes else "changes",
        "summary": f"Review {tag}: reads well overall. {words(tag, 5)}",
        "findings": findings,
    }


def thinking_text(tag: object) -> str:
    return f"Let me think about {tag}. " + words(tag, 60)


class FakeChatClient:
    """Answers every role the generator asks about with valid, never-repeating replies.

    Listing a role in `failing_roles` makes that role raise `LmStudioError` (as an unreachable
    or timed-out server does); `invalid_roles` makes it reply with text that is not JSON.
    """

    def __init__(
        self, failing_roles: Collection[str] = (), invalid_roles: Collection[str] = ()
    ) -> None:
        self.failing_roles = set(failing_roles)
        self.invalid_roles = set(invalid_roles)
        self.roles_called: list[str] = []

    def complete(
        self,
        role: str,
        system: str,
        user: str,
        *,
        thinking: bool,
        schema: Schema | None,
        max_tokens: int,
        temperature: float,
    ) -> Completion:
        self.roles_called.append(role)
        if role in self.failing_roles:
            raise LmStudioError(f"fake LM Studio timed out answering {role}")
        tag = len(self.roles_called)
        content = (
            "this is not json"
            if role in self.invalid_roles
            else getattr(self, f"_{role}")(tag, user)
        )
        return Completion(content, thinking_text(tag) if thinking else "")

    def _scenario(self, tag: int, user: str) -> str:
        profile = re.search(r"Profile: (\S+) -", user)
        assert profile
        return json.dumps(
            {
                "request": f"Build thing {tag} for the {profile[1]} profile. {words(tag)}",
                "existing_files": [SERVICE_PATH, EXAMPLE_PATH, "tests/test_service.py"],
                "example_path": EXAMPLE_PATH,
                "example_content": f"LIMIT_{tag} = 3\nNAME = 'service'\n\n\n"
                f"def limit_{tag}() -> int:\n    return LIMIT_{tag}  # {words(tag)}\n",
                "trivial_task_title": f"Raise limit {tag}",
                "trivial_task_details": f"Change LIMIT_{tag} from 3 to 5.",
                "failure_log": f"FAILED tests/test_service.py::test_it - Error {words(tag)}",
                "question": f"What is the status of job {tag}?",
                "feedback": [
                    {
                        "stage": "spec",
                        "approved": False,
                        "feedback": f"Use smaller functions {words(tag)}",
                    }
                ],
            }
        )

    def _spec(self, tag: int, user: str) -> str:
        return json.dumps(make_spec(tag))

    def _plan(self, tag: int, user: str) -> str:
        return json.dumps(make_plan(tag, re.findall(r"\*\*((?:AC|EC)\d+)\*\*", user)))

    def _diff(self, tag: int, user: str) -> str:
        return f"diff --git a/{SERVICE_PATH} b/{SERVICE_PATH}\n+++ b/{SERVICE_PATH}\n+x = {tag}\n"

    def _review(self, tag: int, user: str) -> str:
        return json.dumps(make_review(tag, passes=tag % 2 == 0))

    def _evidence(self, tag: int, user: str) -> str:
        ids = re.findall(r'"id":\s*"((?:AC|EC)\d+)"', user)
        test = json.loads(user.partition("# Tests that ran\n")[2])[0]
        return json.dumps(
            {
                "criteria": [
                    {"id": i, "tests": [test], "notes": f"Checked {tag}"}
                    for i in ids
                    if i[0] == "A"
                ],
                "edge_cases": [
                    {"id": i, "handling": f"Raises {tag}", "tests": [test]}
                    for i in ids
                    if i[0] == "E"
                ],
            }
        )

    def _commit_message(self, tag: int, user: str) -> str:
        return f"feat(app): add behaviour {tag}\n\n- {words(tag, 6)}\n- {words(tag, 7)}"

    def _pr_body(self, tag: int, user: str) -> str:
        return (
            f"## What changed\n\nAdds behaviour {tag}. {words(tag, 30)}\n\n"
            "## Verification\n\nAll green."
        )

    def _trivial_edit(self, tag: int, user: str) -> str:
        limit = re.search(r"LIMIT_(\d+) = 3", user)
        assert limit
        edit = {
            "path": EXAMPLE_PATH,
            "find": f"LIMIT_{limit[1]} = 3",
            "replace": f"LIMIT_{limit[1]} = 5",
        }
        return json.dumps({"edits": [edit], "note": f"Raised it {tag}"})

    def _docs_pass(self, tag: int, user: str) -> str:
        limit = re.search(r"LIMIT_(\d+) = 3", user)
        assert limit
        find = f"LIMIT_{limit[1]} = 3"
        edit = {"path": EXAMPLE_PATH, "find": find, "replace": f"# Maximum tries {tag}.\n{find}"}
        return json.dumps({"edits": [edit], "note": f"Commented {tag}"})

    def _summarize(self, tag: int, user: str) -> str:
        return f"tests/test_service.py:12 fails with AssertionError because of {words(tag, 8)}"

    def _detect_profile(self, tag: int, user: str) -> str:
        profile = re.search(r"for the ([\w-]+) profile", user.partition("# Profiles")[0])
        assert profile
        return json.dumps({"profile": profile[1], "why": f"The request fits it {tag}"})

    def _memory(self, tag: int, user: str) -> str:
        return json.dumps(
            {
                "you": [f"Prefers small functions {words(tag, 5)}"],
                "decisions": [f"Use uv {words(tag, 5)}"],
            }
        )


# ---- logs shaped like factory-log-samples --------------------------------------------


def logged_call(
    system: str,
    user: str,
    reply: str,
    *,
    reasoning: str = "",
    schema: Schema | None = None,
    thinking: bool = False,
) -> dict[str, Any]:
    """One `llm/NNNN-<role>.json` file's content, built the way the factory writes it."""
    request: dict[str, Any] = {
        "model": "qwen3.6-35b-a3b-mlx",
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    if not thinking:
        request["reasoning_effort"] = "none"
    if schema and not thinking:
        request["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "x", "strict": True, "schema": schema},
        }
    return {"request": request, "reasoning": reasoning, "reply": reply, "usage": {}, "ms": 1}


def event(kind: str, stage: str = "", **data: Any) -> dict[str, Any]:
    return {"kind": kind, "stage": stage, "message": kind, "level": "info", "data": data or None}


def write_job(
    runs_dir: Path, name: str, events: list[dict[str, Any]], calls: dict[str, dict[str, Any]]
) -> Path:
    """Write `runs/<name>/events.jsonl` and `runs/<name>/llm/<file stem>.json` for each call."""
    job_dir = runs_dir / name
    (job_dir / "llm").mkdir(parents=True)
    (job_dir / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    for stem, call in calls.items():
        (job_dir / "llm" / f"{stem}.json").write_text(json.dumps(call))
    return job_dir


def reasoning_call(
    prompts_dir: Path,
    role: str,
    user: str,
    value: dict[str, Any] | str,
    schema: Schema,
    tag: object,
) -> dict[str, Any]:
    """A logged call of a reasoning role: schema appended to the system prompt, reasoning kept."""
    system = guided_system(render_prompt(prompts_dir, f"{role}_system"), schema)
    reply = value if isinstance(value, str) else json.dumps(value)
    return logged_call(system, user, reply, reasoning=thinking_text(tag), thinking=True)


def good_job_calls(prompts_dir: Path, tag: object) -> dict[str, dict[str, Any]]:
    """Spec, plan and review calls for one coherent job, like `sample-good`."""
    spec = make_spec(tag)
    spec_user = render_prompt(
        prompts_dir,
        "spec_user",
        request=f"Build thing {tag}",
        profile="A plain Python package.",
        conventions="Be clear.",
        repo="(new project)",
        memory="(nothing learned yet)",
        previous="none",
        feedback="none",
        references="none",
    )
    plan = make_plan(tag, ["AC1", "AC2", "AC3", "EC1", "EC2"])
    plan_user = render_prompt(
        prompts_dir,
        "plan_user",
        spec=spec_markdown(spec),
        repo="(new project)",
        relevant=[],
        conventions="Be clear.",
        notes={},
        feedback="none",
        references="none",
    )
    review_user = render_prompt(
        prompts_dir,
        "review_user",
        spec=spec_markdown(spec),
        conventions="Be clear.",
        rubric="Be strict.",
        diff=f"diff --git a/{SERVICE_PATH} b/{SERVICE_PATH}\n+x = 1\n{plan_markdown(plan)}",
    )
    return {
        "0001-spec": reasoning_call(prompts_dir, "spec", spec_user, spec, SPEC_SCHEMA, tag),
        "0002-plan": reasoning_call(prompts_dir, "plan", plan_user, plan, PLAN_SCHEMA, tag),
        "0003-review": reasoning_call(
            prompts_dir, "review", review_user, make_review(tag, passes=False), REVIEW_SCHEMA, tag
        ),
        "0004-commit_message": logged_call(
            "Write a commit message.",
            f"Changes {tag}",
            f"feat(app): add behaviour {tag}\n\n- {words(tag, 6)}",
        ),
    }
