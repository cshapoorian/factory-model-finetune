"""Builds the dataset: harvest the factory's logs, then generate synthetic examples until the
token budget is spent. Safe to stop and rerun; it continues where it stopped."""

import itertools
import json
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

from factory_model_finetune.data_formatter import (
    Candidate,
    Example,
    final_answer,
    to_jsonl_line,
    to_messages,
)
from factory_model_finetune.errors import ConfigError, FinetuneError, ResumeError
from factory_model_finetune.lm_client import ChatClient
from factory_model_finetune.log_harvester import harvest_job, job_directories
from factory_model_finetune.progress_reporter import ProgressReporter
from factory_model_finetune.quality_filter import DuplicateIndex, failed_filter
from factory_model_finetune.resume import load_state, save_state
from factory_model_finetune.roles import ALL_ROLES, DEFAULT_ROLE_MIX, PROFILES
from factory_model_finetune.synthetic_generator import SyntheticGenerator
from factory_model_finetune.token_counter import TokenCounter
from factory_model_finetune.upload_guide import render_upload_guide

logger = logging.getLogger(__name__)

TRAIN_FILE = "train.jsonl"
EVAL_FILE = "eval.jsonl"
STATE_FILE = "state.json"
UPLOAD_FILE = "UPLOAD.md"
# A long run of scenarios that yield nothing means something is wrong (a prompt the model cannot
# answer, a broken server); stopping beats spinning all night.
MAX_SCENARIOS_WITHOUT_PROGRESS = 50


@dataclass(frozen=True)
class BuildConfig:
    """Everything a build needs to know; validated on creation."""

    factory_runs: Path
    out_dir: Path
    prompts_dir: Path
    budget_usd: float = 15.0
    price_per_million: float = 2.10
    epochs: int = 1
    role_mix: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_ROLE_MIX))
    eval_fraction: float = 0.03
    seed: int = 0

    def __post_init__(self) -> None:
        if self.budget_usd <= 0 or self.price_per_million <= 0:
            raise ConfigError(
                f"budget ({self.budget_usd}) and price per million ({self.price_per_million}) "
                "must both be positive"
            )
        if self.epochs < 1:
            raise ConfigError(f"epochs must be at least 1, got {self.epochs}")
        if set(self.role_mix) != set(ALL_ROLES) or min(self.role_mix.values()) <= 0:
            raise ConfigError(
                f"role mix must give each of {ALL_ROLES} a positive weight, got {self.role_mix}"
            )
        if not 0 <= self.eval_fraction < 0.5:
            raise ConfigError(f"eval fraction must be in [0, 0.5), got {self.eval_fraction}")
        if not self.prompts_dir.is_dir():
            raise ConfigError(f"prompts folder {self.prompts_dir} not found; set --prompts-dir")

    @property
    def target_tokens(self) -> int:
        """Dataset tokens that spend the budget: billed tokens are dataset tokens times epochs."""
        billed_tokens = self.budget_usd / self.price_per_million * 1_000_000
        return int(billed_tokens / self.epochs)


@dataclass(frozen=True)
class BuildReport:
    """What a build produced."""

    target_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    examples_written: int
    eval_examples_written: int
    tokens_by_role: dict[str, int]
    tokens_by_source: dict[str, int]

    def format(self) -> str:
        """Return the report as text for the terminal."""
        lines = [
            f"Train tokens: {self.total_tokens:,} of {self.target_tokens:,} target",
            f"Estimated cost: ${self.estimated_cost_usd:.2f}",
            f"Examples: {self.examples_written} train, {self.eval_examples_written} eval",
            "By role: " + _share_text(self.tokens_by_role, self.total_tokens),
            "By source: " + _share_text(self.tokens_by_source, self.total_tokens),
        ]
        return "\n".join(lines)


class DatasetBuilder:
    """Writes `train.jsonl`, `eval.jsonl`, the resume file and the upload guide into one folder."""

    def __init__(self, config: BuildConfig, counter: TokenCounter) -> None:
        """Open (and resume, if it already has data) the dataset folder `config.out_dir`.

        Raises:
            ResumeError: an existing output file cannot be read back.
        """
        self._config = config
        self._counter = counter
        config.out_dir.mkdir(parents=True, exist_ok=True)
        self._train_path = config.out_dir / TRAIN_FILE
        self._eval_path = config.out_dir / EVAL_FILE
        self._state_path = config.out_dir / STATE_FILE
        for path in (self._train_path, self._eval_path):
            _drop_partial_last_line(path)
        self._state = load_state(self._state_path)
        self._index = DuplicateIndex()
        self._remember_written_examples()
        self._quotas = _role_quotas(config.target_tokens, config.role_mix)
        self._reporter = ProgressReporter(config.target_tokens, self._state.total_tokens)

    def build(self, client: ChatClient) -> BuildReport:
        """Harvest logs, then generate until the budget is spent; return the report.

        State is saved after every example, so an interrupt (or an error) loses nothing.
        """
        try:
            self._harvest()
            self._generate(client)
        finally:
            save_state(self._state_path, self._state)
        (self._config.out_dir / UPLOAD_FILE).write_text(
            render_upload_guide(
                self._config.out_dir,
                self._state.total_tokens,
                self._config.epochs,
                self._config.price_per_million,
            )
        )
        return self._report()

    def accept(self, candidate: Candidate) -> bool:
        """Apply the quality filters and write the example; return whether it passed them.

        True means the answer is good, not that it was saved: a candidate whose role already has
        its share of the budget passes without being written. The generator relies on that, since
        the next role in a chain (spec, then plan, then review) is built from the answer itself and
        stays worth training on. Such candidates still count as accepted in the progress report.
        """
        if self._budget_is_spent():
            return False
        if rejected_by := failed_filter(candidate, self._index):
            self._reporter.rejected(rejected_by)
            return False
        if self._state.tokens_by_role.get(candidate.role, 0) < self._quotas[candidate.role]:
            self._write(candidate)
        else:
            self._reporter.accepted(0)
        return True

    # ---- steps ------------------------------------------------------------------------

    def _harvest(self) -> None:
        for job_dir in job_directories(self._config.factory_runs):
            if self._budget_is_spent():
                return
            if self._state.last_job_id and job_dir.name <= self._state.last_job_id:
                continue
            for candidate in harvest_job(job_dir, self._reporter):
                self.accept(candidate)
            self._state.last_job_id = job_dir.name
            save_state(self._state_path, self._state)
            logger.info("harvested job %s | %s", job_dir.name, self._reporter.summary_line())

    def _generate(self, client: ChatClient) -> None:
        rng = random.Random(self._config.seed + self._state.examples_written)
        generator = SyntheticGenerator(
            client, self._config.prompts_dir, self._reporter, self.accept, rng
        )
        first_profile = rng.randrange(len(PROFILES))
        scenarios_without_progress = 0
        for number in itertools.count():
            if self._budget_is_spent():
                return
            written_before = self._state.examples_written + self._state.eval_examples_written
            open_roles = {r for r, quota in self._quotas.items() if self._tokens_of(r) < quota}
            generator.run_scenario(PROFILES[(first_profile + number) % len(PROFILES)], open_roles)
            logger.info("scenario %d | %s", number + 1, self._reporter.summary_line())
            made_progress = (
                self._state.examples_written + self._state.eval_examples_written > written_before
            )
            scenarios_without_progress = 0 if made_progress else scenarios_without_progress + 1
            if scenarios_without_progress >= MAX_SCENARIOS_WITHOUT_PROGRESS:
                raise FinetuneError(
                    f"{MAX_SCENARIOS_WITHOUT_PROGRESS} scenarios in a row produced no example; "
                    f"rejections so far: {self._reporter.rejection_rates()}"
                )

    # ---- writing ------------------------------------------------------------------------

    def _write(self, candidate: Candidate) -> None:
        messages = to_messages(candidate)
        example = Example(
            candidate.role, candidate.source, messages, self._counter.count_messages(messages)
        )
        is_eval = self._next_is_eval(candidate.role)
        path = self._eval_path if is_eval else self._train_path
        with path.open("a") as file:
            file.write(to_jsonl_line(example.messages) + "\n")
        self._index.add(candidate.system, candidate.user, candidate.reply)
        if is_eval:
            self._state.eval_examples_written += 1
            counts = self._state.eval_examples_by_role
            counts[candidate.role] = counts.get(candidate.role, 0) + 1
            self._reporter.accepted(0)  # eval tokens are not billed, so they don't count
        else:
            self._record_train_example(example)
        save_state(self._state_path, self._state)

    def _record_train_example(self, example: Example) -> None:
        state = self._state
        state.total_tokens += example.tokens
        state.examples_written += 1
        state.train_examples_by_role[example.role] = (
            state.train_examples_by_role.get(example.role, 0) + 1
        )
        for counts, key in (
            (state.tokens_by_role, example.role),
            (state.tokens_by_source, example.source),
        ):
            counts[key] = counts.get(key, 0) + example.tokens
        self._reporter.accepted(example.tokens)

    def _remember_written_examples(self) -> None:
        """Feed earlier examples to the duplicate check so a resumed run cannot repeat them."""
        for path in (self._train_path, self._eval_path):
            if not path.exists():
                continue
            for number, line in enumerate(path.read_text().splitlines(), start=1):
                try:
                    system, user, assistant = json.loads(line)["messages"]
                    self._index.add(
                        system["content"], user["content"], final_answer(assistant["content"])
                    )
                except (json.JSONDecodeError, KeyError, ValueError, TypeError) as error:
                    raise ResumeError(
                        f"{path} line {number} is not a valid example: {error!r}"
                    ) from error

    def _next_is_eval(self, role: str) -> bool:
        """Decide if the next example for `role` belongs in the eval set.

        Uses a deterministic, resume-safe threshold: `ceil(eval_fraction * total_seen)`.
        This ensures every role, including rare ones, eventually contributes to evaluation
        without requiring a global shuffle or pre-calculation. The threshold is checked against
        `seen + 1` to ensure the very first example for a role is always considered for eval
        if the fraction allows it (e.g., 0.03 * 1 = 0.03, ceil is 1).
        """
        state = self._state
        eval_count = state.eval_examples_by_role.get(role, 0)
        seen = state.train_examples_by_role.get(role, 0) + eval_count
        return eval_count < math.ceil(self._config.eval_fraction * (seen + 1))

    # ---- bookkeeping ------------------------------------------------------------------------

    def _budget_is_spent(self) -> bool:
        return self._state.total_tokens >= self._config.target_tokens

    def _tokens_of(self, role: str) -> int:
        return self._state.tokens_by_role.get(role, 0)

    def _report(self) -> BuildReport:
        state = self._state
        billed_tokens = state.total_tokens * self._config.epochs
        return BuildReport(
            target_tokens=self._config.target_tokens,
            total_tokens=state.total_tokens,
            estimated_cost_usd=billed_tokens / 1_000_000 * self._config.price_per_million,
            examples_written=state.examples_written,
            eval_examples_written=state.eval_examples_written,
            tokens_by_role=dict(state.tokens_by_role),
            tokens_by_source=dict(state.tokens_by_source),
        )


def _role_quotas(target_tokens: int, role_mix: dict[str, float]) -> dict[str, float]:
    total_weight = sum(role_mix.values())
    return {role: target_tokens * weight / total_weight for role, weight in role_mix.items()}


def _drop_partial_last_line(path: Path) -> None:
    """Remove a last line cut short by a crash, so appending never glues two examples together."""
    if not path.exists():
        return
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return
    path.write_bytes(data[: data.rfind(b"\n") + 1])
    logger.warning("dropped an incomplete last line from %s", path)


def _share_text(tokens_by_key: dict[str, int], total: int) -> str:
    if not total:
        return "none"
    ranked = sorted(tokens_by_key.items(), key=lambda item: -item[1])
    return ", ".join(f"{key} {tokens:,} ({tokens / total:.0%})" for key, tokens in ranked)
