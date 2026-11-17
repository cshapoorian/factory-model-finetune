"""Makes synthetic training examples: the local model answers the factory's real prompts.

Each run invents one scenario (a request, a small repo, a failing gate log, ...) for a profile,
then builds every role's real prompt from it. Roles that depend on an earlier one (plan needs a
spec, review needs a diff) are fed the earlier role's answer, but only if that answer was
accepted. Every answer goes through `accept`, which applies the quality filters.
"""

import functools
import json
import logging
import random
import re
from collections.abc import Callable, Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from factory_model_finetune.data_formatter import Candidate
from factory_model_finetune.errors import LmStudioError
from factory_model_finetune.factory_text import (
    defang_think,
    evidence_markdown,
    guided_system,
    plan_markdown,
    render_prompt,
    spec_markdown,
)
from factory_model_finetune.lm_client import ChatClient, Completion
from factory_model_finetune.profiles import PROFILE_INFO, Profile
from factory_model_finetune.progress_reporter import ProgressReporter
from factory_model_finetune.quality_filter import CONSISTENCY
from factory_model_finetune.roles import PROFILES, REASONING_ROLES
from factory_model_finetune.schemas import (
    EDITS_SCHEMA,
    MAPPING_SCHEMA,
    MEMORY_SCHEMA,
    ROLE_SCHEMAS,
    SCHEMA_FILTER,
    STR,
    STRS,
    Schema,
    detect_profile_schema,
    object_schema,
    valid_reply,
)

logger = logging.getLogger(__name__)

SOURCE = "synthetic"
LM_ERROR = "lm_error"
BAD_SCENARIO = "scenario"
MAX_CONSECUTIVE_LM_ERRORS = 20
ANSWER_TEMPERATURE = 0.2  # the factory's own default, so answers look like production ones
INVENTION_TEMPERATURE = 1.0  # scenarios should vary a lot between runs
NO_MEMORY = "(nothing learned yet)"
CHAIN_ROLES = frozenset({"spec", "plan", "review", "evidence", "commit_message", "pr_body"})
MAX_TOKENS = {
    "spec": 16000,
    "plan": 16000,
    "review": 8000,
    "summarize": 2000,
    "commit_message": 400,
    "pr_body": 3000,
    "diff": 6000,
}
DEFAULT_MAX_TOKENS = 8192

THEMES = (
    "a small-business invoicing tool",
    "a personal finance tracker",
    "a recipe and meal-planning app",
    "an inventory system for a bike shop",
    "a habit and fitness tracker",
    "a support-ticket triage service",
    "a library or book-lending catalogue",
    "an event booking and ticketing system",
    "a log analyser for a web server",
    "a CSV cleaning and reporting tool",
    "a weather and sensor dashboard",
    "a customer churn analysis",
    "an A/B test evaluation",
    "a document search assistant",
    "a chatbot that answers questions about internal docs",
    "a feature-flag service",
    "a job scheduler and queue monitor",
    "a URL shortener",
    "a markdown note-taking app",
    "a team standup bot",
    "a price-scraping and alerting pipeline",
    "a pull-request review summariser",
    "a clinic appointment scheduler",
    "a classroom grading helper",
)
REQUEST_STYLES = (
    "a one-line ask for a brand-new project",
    "a detailed paragraph with explicit requirements for a new project",
    "a small feature to add to an existing codebase",
    "a bug report with symptoms for an existing codebase",
    "a refactor of messy existing code",
    "a request to add tests and error handling to existing code",
    "a vague idea the person has not thought through",
    "a request with a clear constraint (deadline, library, or data size)",
)

SCENARIO_SYSTEM = (
    "You invent realistic inputs for testing a software factory that builds projects for a "
    "person. Be concrete and varied. Never mention that the input is synthetic."
)
SCENARIO_INSTRUCTIONS = """Invent one scenario for the profile above. Reply with JSON containing:
- request: what the person types to the factory (1-6 sentences, natural wording).
- existing_files: file paths already in the repository (an empty list for a brand-new project, \
otherwise 6-15 realistic paths for this profile).
- example_path, example_content: one existing source file (30-60 lines of real code in the \
profile's language) that fits the project.
- trivial_task_title, trivial_task_details: a tiny mechanical edit to example_content (rename a \
constant, change a config value, fix a typo), described in 1-3 sentences.
- failure_log: realistic failing output from this profile's quality gates (test failures, lint \
or type errors), 15-40 lines.
- question: something the person might ask about the job, as a short chat message.
- feedback: 1-3 notes the person gave while reviewing the job's stages. Each has stage (spec, \
plan or review), approved (true or false) and feedback (a specific preference or correction, in \
their own words)."""
SCENARIO_SCHEMA = object_schema(
    {
        "request": STR,
        "existing_files": STRS,
        "example_path": STR,
        "example_content": STR,
        "trivial_task_title": STR,
        "trivial_task_details": STR,
        "failure_log": STR,
        "question": STR,
        "feedback": {
            "type": "array",
            "items": object_schema(
                {"stage": STR, "approved": {"type": "boolean"}, "feedback": STR}
            ),
        },
    }
)
DIFF_SYSTEM = (
    "You are an engineer implementing a plan. Reply with only a git unified diff (with "
    "`diff --git` headers) that implements the work, including its tests. No explanation."
)
FLAWED_DIFF_HINT = (
    "Implement it with one realistic defect a careful reviewer should catch (for example a "
    "missed edge case, a swallowed error, a missing test, or a convention violation). "
    "Do not mention the defect."
)
CLEAN_DIFF_HINT = "Implement it faithfully and cleanly, following the conventions."
FIXED_TEST = "tests.test_smoke::test_package_imports"
JOB_STATES = (("done", "pr"), ("awaiting_approval", "spec"), ("needs_you", "implement"))


@dataclass(frozen=True)
class Scenario:
    """The invented material one run builds its prompts from."""

    profile: str
    request: str
    existing_files: list[str]
    example_path: str
    example_content: str
    trivial_task_title: str
    trivial_task_details: str
    failure_log: str
    question: str
    feedback: list[dict[str, Any]]


class SyntheticGenerator:
    """Runs scenarios against a local model and hands each answer to `accept`."""

    def __init__(
        self,
        client: ChatClient,
        prompts_dir: Path,
        reporter: ProgressReporter,
        accept: Callable[[Candidate], bool],
        rng: random.Random,
    ) -> None:
        """`accept` applies the quality filters and returns whether the candidate passed them."""
        self._client = client
        self._prompt = functools.partial(render_prompt, prompts_dir)
        self._reporter = reporter
        self._accept = accept
        self._rng = rng
        self._lm_errors_in_a_row = 0

    def run_scenario(self, profile_name: str, needed_roles: Collection[str]) -> None:
        """Invent a scenario for `profile_name` and answer the roles in `needed_roles`.

        Raises:
            LmStudioError: the model failed too many calls in a row to keep going.
        """
        scenario = self._invent_scenario(PROFILE_INFO[profile_name])
        if scenario is None:
            return
        independent_steps = {
            "trivial_edit": self._trivial_edit,
            "docs_pass": self._docs_pass,
            "summarize": self._summarize,
            "detect_profile": self._detect_profile,
            "memory": self._memory,
        }
        for role, step in independent_steps.items():
            if role in needed_roles:
                self._produced(step(scenario))
        wanted = CHAIN_ROLES & set(needed_roles)
        if wanted:
            self._run_chain(scenario, wanted)

    # ---- roles that build on each other ------------------------------------------

    def _run_chain(self, scenario: Scenario, wanted: frozenset[str]) -> None:
        spec = self._produced(self._spec(scenario))
        if spec is None or wanted == {"spec"}:
            return
        plan = self._produced(self._plan(scenario, spec))
        if plan is None:
            return
        if "review" in wanted:
            self._review(scenario, spec, plan)
        if "commit_message" in wanted:
            self._produced(self._commit_message(spec, plan))
        if wanted & {"evidence", "pr_body"}:
            tests = self._tests_that_ran(plan)
            mapping = self._produced(self._evidence(spec, tests))
            if mapping is not None and "pr_body" in wanted:
                self._produced(self._pr_body(spec, mapping, tests))

    def _spec(self, scenario: Scenario) -> Candidate | None:
        profile = PROFILE_INFO[scenario.profile]
        user = self._prompt(
            "spec_user",
            request=scenario.request,
            profile=profile.description,
            conventions=profile.conventions,
            repo=_repo_overview(scenario),
            memory=NO_MEMORY,
            previous="none",
            feedback="none",
            references="none",
        )
        return self._answer("spec", self._prompt("spec_system"), user)

    def _plan(self, scenario: Scenario, spec: dict[str, Any]) -> Candidate | None:
        files = scenario.existing_files
        user = self._prompt(
            "plan_user",
            spec=spec_markdown(spec),
            repo=_repo_overview(scenario),
            relevant=self._rng.sample(files, k=min(5, len(files))),
            conventions=PROFILE_INFO[scenario.profile].conventions,
            notes={},
            feedback="none",
            references="none",
        )
        return self._answer("plan", self._prompt("plan_system"), user)

    def _review(self, scenario: Scenario, spec: dict[str, Any], plan: dict[str, Any]) -> None:
        profile = PROFILE_INFO[scenario.profile]
        diff = self._write_diff(spec, plan)
        if diff is None:
            return
        user = self._prompt(
            "review_user",
            spec=spec_markdown(spec),
            conventions=profile.conventions,
            rubric=profile.rubric,
            diff=diff,
        )
        self._produced(self._answer("review", self._prompt("review_system"), user))

    def _write_diff(self, spec: dict[str, Any], plan: dict[str, Any]) -> str | None:
        hint = FLAWED_DIFF_HINT if self._rng.random() < 0.5 else CLEAN_DIFF_HINT
        user = f"{spec_markdown(spec)}\n\n{plan_markdown(plan)}\n\n{hint}"
        completion = self._ask("diff", DIFF_SYSTEM, user, thinking=False, schema=None)
        if completion is None:
            return None
        diff = _without_code_fence(completion.content)
        if "diff --git" not in diff:
            self._reporter.rejected(BAD_SCENARIO)
            return None
        return diff

    def _commit_message(self, spec: dict[str, Any], plan: dict[str, Any]) -> Candidate | None:
        files = sorted({f for task in plan["tasks"] for f in task["files"]})
        numstat = "\n".join(
            f"{self._rng.randint(1, 120)}\t{self._rng.randint(0, 40)}\t{f}" for f in files
        )
        user = f"{spec_markdown(spec)}\n\nChanged files (added/removed/path):\n{numstat}"
        return self._answer("commit_message", self._prompt("commit_system"), user)

    def _tests_that_ran(self, plan: dict[str, Any]) -> list[str]:
        """Names of the tests a run would list: most planned ones, plus an unrelated old test."""
        planned = [
            f"tests.test_{task['id'].lower()}::{name}"
            for task in plan["tasks"]
            for name in task["tests"]
        ]
        kept = [name for name in planned if self._rng.random() < 0.85]
        return (kept or planned) + [FIXED_TEST]

    def _evidence(self, spec: dict[str, Any], tests: list[str]) -> Candidate | None:
        user = self._prompt(
            "evidence_user",
            criteria=spec["acceptance_criteria"],
            edge_cases=spec["edge_cases"],
            tests=tests,
        )
        return self._answer("evidence", self._prompt("evidence_system"), user, MAPPING_SCHEMA)

    def _pr_body(
        self, spec: dict[str, Any], mapping: dict[str, Any], tests: list[str]
    ) -> Candidate | None:
        user = self._prompt(
            "pr_user",
            spec=spec_markdown(spec),
            evidence=evidence_markdown(spec["title"], spec, mapping, tests),
            open_findings="none",
        )
        return self._answer("pr_body", self._prompt("pr_system"), user)

    # ---- roles that stand alone ----------------------------------------------------

    def _trivial_edit(self, scenario: Scenario) -> Candidate | None:
        task = {
            "id": "T1",
            "title": scenario.trivial_task_title,
            "details": scenario.trivial_task_details,
            "files": [scenario.example_path],
            "kind": "trivial",
            "tests": [],
            "covers": [],
        }
        user = self._prompt(
            "trivial_edit_user",
            task=task,
            files={scenario.example_path: scenario.example_content},
            conventions=PROFILE_INFO[scenario.profile].conventions,
        )
        return self._answer("trivial_edit", self._prompt("trivial_edit_system"), user, EDITS_SCHEMA)

    def _docs_pass(self, scenario: Scenario) -> Candidate | None:
        user = self._prompt(
            "docs_user",
            path=scenario.example_path,
            content=scenario.example_content,
            conventions=PROFILE_INFO[scenario.profile].conventions,
        )
        return self._answer("docs_pass", self._prompt("docs_system"), user, EDITS_SCHEMA)

    def _summarize(self, scenario: Scenario) -> Candidate | None:
        """The factory's `summarize` role summarises gate failures and answers the person."""
        if self._rng.random() < 0.5:
            return self._answer("summarize", self._prompt("summarize_system"), scenario.failure_log)
        user = self._prompt(
            "answer_user", context=self._job_context(scenario), question=scenario.question
        )
        return self._answer("summarize", self._prompt("answer_system"), user)

    def _job_context(self, scenario: Scenario) -> dict[str, Any]:
        status, stage = self._rng.choice(JOB_STATES)
        job_id = f"{self._rng.getrandbits(32):08x}"
        return {
            "id": job_id,
            "title": scenario.request[:60],
            "request": scenario.request,
            "status": status,
            "stage": stage,
            "profile": scenario.profile,
            "error": scenario.failure_log.splitlines()[0][:200] if status == "needs_you" else None,
            "goal": None,
            "evidence": None,
            "pr_url": None,
            "branch": f"factory/{job_id}",
            "worktree": f"/work/.worktrees/{job_id}",
            "preview_url": None,
            "recent_events": [f"[{stage}] stage: ▶ {stage}"],
            "thread": [f"you: {scenario.request[:200]}"],
            "artifacts_dir": f"/work/runs/{job_id}",
        }

    def _detect_profile(self, scenario: Scenario) -> Candidate | None:
        others = [name for name in PROFILES if name != scenario.profile]
        options: list[str]
        if self._rng.random() < 0.6:
            options = list(PROFILES)
        else:
            chosen = {scenario.profile, *self._rng.sample(others, k=self._rng.randint(1, 2))}
            options = [name for name in PROFILES if name in chosen]
        user = self._prompt(
            "profile_user",
            request=scenario.request,
            profiles={name: PROFILE_INFO[name].description for name in options},
            repo=_repo_overview(scenario),
        )
        candidate = self._answer(
            "detect_profile", self._prompt("profile_system"), user, detect_profile_schema(options)
        )
        if candidate and json.loads(candidate.reply)["profile"] != scenario.profile:
            self._reporter.rejected(CONSISTENCY)  # the scenario was written for a known profile
            return None
        return candidate

    def _memory(self, scenario: Scenario) -> Candidate | None:
        user = self._prompt(
            "memory_user", memory=NO_MEMORY, request=scenario.request, feedback=scenario.feedback
        )
        return self._answer("memory", self._prompt("memory_system"), user, MEMORY_SCHEMA)

    # ---- talking to the model --------------------------------------------------------

    def _invent_scenario(self, profile: Profile) -> Scenario | None:
        theme, style = self._rng.choice(THEMES), self._rng.choice(REQUEST_STYLES)
        user = (
            f"Profile: {profile.name} - {profile.description}\n"
            f"Domain: {theme}\nRequest style: {style}\n\n{SCENARIO_INSTRUCTIONS}"
        )
        completion = self._ask(
            "scenario", SCENARIO_SYSTEM, user, thinking=False, schema=SCENARIO_SCHEMA
        )
        reply = valid_reply(completion.content, SCENARIO_SCHEMA) if completion else None
        if reply is None:
            self._reporter.rejected(BAD_SCENARIO)
            return None
        scenario = Scenario(profile=profile.name, **json.loads(reply))
        if len(scenario.request) < 30 or len(scenario.example_content) < 100:
            self._reporter.rejected(BAD_SCENARIO)
            return None
        return scenario

    def _answer(
        self, role: str, system: str, user: str, schema: Schema | None = None
    ) -> Candidate | None:
        """Ask for one role's answer in the form the factory would; None if it is unusable.

        Reasoning roles get their schema in the prompt and reasoning on, as in the factory.
        """
        schema = schema or ROLE_SCHEMAS.get(role)
        thinking = role in REASONING_ROLES
        if thinking and schema:
            system = guided_system(system, schema)
        system, user = defang_think(system), defang_think(user)
        completion = self._ask(role, system, user, thinking=thinking, schema=schema)
        if completion is None:
            return None
        reply = valid_reply(completion.content, schema)
        if reply is None:
            self._reporter.rejected(SCHEMA_FILTER)
            return None
        reasoning = completion.reasoning.strip() if thinking else ""
        return Candidate(role, SOURCE, system, user, reasoning, reply)

    def _ask(
        self, role: str, system: str, user: str, *, thinking: bool, schema: Schema | None
    ) -> Completion | None:
        """Call the model; on failure log the error and skip, unless failures keep piling up."""
        try:
            completion = self._client.complete(
                role,
                system,
                user,
                thinking=thinking,
                # LM Studio's strict grammar and reasoning don't mix, so reasoning roles go without.
                schema=None if thinking else schema,
                max_tokens=MAX_TOKENS.get(role, DEFAULT_MAX_TOKENS),
                temperature=INVENTION_TEMPERATURE if role == "scenario" else ANSWER_TEMPERATURE,
            )
        except LmStudioError as error:
            logger.error("skipping this example: %s", error)
            self._reporter.rejected(LM_ERROR)
            self._lm_errors_in_a_row += 1
            if self._lm_errors_in_a_row >= MAX_CONSECUTIVE_LM_ERRORS:
                raise LmStudioError(
                    f"{self._lm_errors_in_a_row} LM Studio calls failed in a row; is it running "
                    "with the model loaded?"
                ) from error
            return None
        self._lm_errors_in_a_row = 0
        return completion

    def _produced(self, candidate: Candidate | None) -> Any | None:
        """Offer `candidate` to `accept`; return its parsed answer if it was accepted."""
        if candidate is None or not self._accept(candidate):
            return None
        return json.loads(candidate.reply) if candidate.role in ROLE_SCHEMAS else candidate.reply


def _repo_overview(scenario: Scenario) -> str:
    if not scenario.existing_files:
        return "(new project)"
    return "Files:\n" + "\n".join(scenario.existing_files)


def _without_code_fence(text: str) -> str:
    return re.sub(r"^```\w*\n|\n```\s*$", "", text.strip())
