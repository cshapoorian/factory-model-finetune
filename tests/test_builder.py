import json
from pathlib import Path

import pytest
from fakes import FakeChatClient, good_job_calls, make_spec, thinking_text, write_job

from factory_model_finetune.builder import BuildConfig, BuildReport, DatasetBuilder
from factory_model_finetune.data_formatter import Candidate
from factory_model_finetune.errors import ConfigError
from factory_model_finetune.lm_client import Completion
from factory_model_finetune.roles import ALL_ROLES, REASONING_ROLES
from factory_model_finetune.schemas import Schema
from factory_model_finetune.token_counter import TokenCounter

# 0.01 dollars at 2.10 per million is about 4,761 tokens: a few dozen fake-tokenizer examples.
SMALL_BUDGET = 0.01


class InterruptingClient(FakeChatClient):
    """Behaves like the fake until `calls_allowed` calls are used, then simulates Ctrl-C."""

    def __init__(self, calls_allowed: int) -> None:
        super().__init__()
        self._calls_allowed = calls_allowed

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
        if len(self.roles_called) >= self._calls_allowed:
            raise KeyboardInterrupt
        return super().complete(
            role,
            system,
            user,
            thinking=thinking,
            schema=schema,
            max_tokens=max_tokens,
            temperature=temperature,
        )


def make_config(tmp_path: Path, prompts_dir: Path, **overrides: object) -> BuildConfig:
    settings: dict[str, object] = {
        "factory_runs": tmp_path / "runs",
        "out_dir": tmp_path / "data",
        "prompts_dir": prompts_dir,
        "budget_usd": SMALL_BUDGET,
        "eval_fraction": 0.2,
    }
    return BuildConfig(**{**settings, **overrides})  # type: ignore[arg-type]


def read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def build(config: BuildConfig, counter: TokenCounter, client: FakeChatClient) -> BuildReport:
    return DatasetBuilder(config, counter).build(client)


def test_target_tokens_follow_budget_price_and_epochs(tmp_path: Path, prompts_dir: Path):
    default = make_config(tmp_path, prompts_dir, budget_usd=15.0)
    two_epochs = make_config(tmp_path, prompts_dir, budget_usd=15.0, epochs=2)

    assert default.target_tokens == 7_142_857
    assert two_epochs.target_tokens == 3_571_428


@pytest.mark.parametrize(
    "overrides",
    [
        {"budget_usd": 0},
        {"price_per_million": -1},
        {"epochs": 0},
        {"eval_fraction": 0.9},
        {"role_mix": {"spec": 1.0}},
    ],
)
def test_rejects_invalid_settings(tmp_path: Path, prompts_dir: Path, overrides: dict):
    with pytest.raises(ConfigError):
        make_config(tmp_path, prompts_dir, **overrides)


def test_respects_token_budget(tmp_path: Path, prompts_dir: Path, counter: TokenCounter):
    config = make_config(tmp_path, prompts_dir)

    report = build(config, counter, FakeChatClient())

    counted = sum(
        counter.count_messages(line["messages"])
        for line in read_lines(config.out_dir / "train.jsonl")
    )
    assert counted == report.total_tokens
    assert report.total_tokens >= config.target_tokens
    assert report.total_tokens < config.target_tokens * 1.5
    assert report.estimated_cost_usd == pytest.approx(
        report.total_tokens / 1_000_000 * config.price_per_million
    )


def test_cost_counts_every_epoch(tmp_path: Path, prompts_dir: Path, counter: TokenCounter):
    config = make_config(tmp_path, prompts_dir, epochs=2)

    report = build(config, counter, FakeChatClient())

    assert report.estimated_cost_usd == pytest.approx(
        report.total_tokens * 2 / 1_000_000 * config.price_per_million
    )
    assert report.estimated_cost_usd >= SMALL_BUDGET * 0.99


def test_role_mix_distribution(tmp_path: Path, prompts_dir: Path, counter: TokenCounter):
    config = make_config(tmp_path, prompts_dir, budget_usd=0.03, eval_fraction=0.0)

    report = build(config, counter, FakeChatClient())

    by_role = report.tokens_by_role
    hard_share = sum(by_role[role] for role in REASONING_ROLES) / report.total_tokens
    assert set(by_role) == set(ALL_ROLES)
    assert hard_share > 0.5
    assert min(by_role[role] for role in REASONING_ROLES) > max(
        by_role[role] for role in ALL_ROLES if role not in REASONING_ROLES
    )


def test_eval_split(tmp_path: Path, prompts_dir: Path, counter: TokenCounter):
    config = make_config(tmp_path, prompts_dir)

    report = build(config, counter, FakeChatClient())

    train = read_lines(config.out_dir / "train.jsonl")
    held_out = read_lines(config.out_dir / "eval.jsonl")
    assert len(train) == report.examples_written
    assert len(held_out) == report.eval_examples_written > 0
    train_prompts = {example["messages"][1]["content"] for example in train}
    assert not train_prompts & {example["messages"][1]["content"] for example in held_out}


def test_eval_split_draws_from_every_role(tmp_path: Path, prompts_dir: Path, counter: TokenCounter):
    config = make_config(tmp_path, prompts_dir, budget_usd=0.03, eval_fraction=0.03)

    build(config, counter, FakeChatClient())

    state = json.loads((config.out_dir / "state.json").read_text())
    assert set(state["eval_examples_by_role"]) == set(ALL_ROLES)


def spec_candidate(tag: int) -> Candidate:
    return Candidate(
        "spec",
        "synthetic",
        "system",
        f"request {tag}",
        thinking_text(tag),
        json.dumps(make_spec(tag)),
    )


def test_accepts_without_writing_when_the_role_quota_is_full(
    tmp_path: Path, prompts_dir: Path, counter: TokenCounter
):
    role_mix = {role: 100.0 for role in ALL_ROLES} | {"spec": 0.001}
    config = make_config(tmp_path, prompts_dir, role_mix=role_mix, eval_fraction=0.0)
    builder = DatasetBuilder(config, counter)

    assert builder.accept(spec_candidate(1))
    assert builder.accept(spec_candidate(2))

    assert len(read_lines(config.out_dir / "train.jsonl")) == 1


def test_writes_upload_guide(tmp_path: Path, prompts_dir: Path, counter: TokenCounter):
    config = make_config(tmp_path, prompts_dir, epochs=3)

    build(config, counter, FakeChatClient())

    guide = (config.out_dir / "UPLOAD.md").read_text()
    assert "firectl signin" in guide
    assert f"firectl dataset create <dataset-name> {config.out_dir / 'train.jsonl'}" in guide
    assert f"{config.out_dir / 'eval.jsonl'}" in guide
    assert "--epochs 3" in guide


def test_harvested_logs_come_first_and_are_marked(
    tmp_path: Path, prompts_dir: Path, counter: TokenCounter
):
    runs = tmp_path / "runs"
    runs.mkdir()
    events = [{"kind": "done", "stage": "", "message": "", "level": "info", "data": None}]
    write_job(runs, "job-1", events, good_job_calls(prompts_dir, 1))
    config = make_config(tmp_path, prompts_dir, eval_fraction=0.0)

    report = build(config, counter, FakeChatClient())

    assert report.tokens_by_source["harvest"] > 0
    assert report.tokens_by_source["synthetic"] > 0


def test_continues_without_logs(tmp_path: Path, prompts_dir: Path, counter: TokenCounter):
    config = make_config(tmp_path, prompts_dir, factory_runs=tmp_path / "missing")

    report = build(config, counter, FakeChatClient())

    assert set(report.tokens_by_source) == {"synthetic"}


def test_stops_mid_generation_and_saves_state(
    tmp_path: Path, prompts_dir: Path, counter: TokenCounter
):
    config = make_config(tmp_path, prompts_dir, eval_fraction=0.0)

    with pytest.raises(KeyboardInterrupt):
        build(config, counter, InterruptingClient(calls_allowed=5))

    state = json.loads((config.out_dir / "state.json").read_text())
    train = read_lines(config.out_dir / "train.jsonl")
    assert len(train) == state["examples_written"] > 0
    assert state["total_tokens"] < config.target_tokens


def test_resumes_without_duplication(tmp_path: Path, prompts_dir: Path, counter: TokenCounter):
    config = make_config(tmp_path, prompts_dir, eval_fraction=0.0)
    with pytest.raises(KeyboardInterrupt):
        build(config, counter, InterruptingClient(calls_allowed=5))
    before = (config.out_dir / "train.jsonl").read_text()

    report = build(config, counter, FakeChatClient())

    after = (config.out_dir / "train.jsonl").read_text()
    lines = after.splitlines()
    assert after.startswith(before)
    assert len(lines) == len(set(lines)) == report.examples_written
    assert report.total_tokens >= config.target_tokens


def test_resume_drops_a_line_cut_short_by_a_crash(
    tmp_path: Path, prompts_dir: Path, counter: TokenCounter
):
    config = make_config(tmp_path, prompts_dir, eval_fraction=0.0)
    with pytest.raises(KeyboardInterrupt):
        build(config, counter, InterruptingClient(calls_allowed=5))
    train = config.out_dir / "train.jsonl"
    train.write_text(train.read_text() + '{"messages": [{"role": "sys')

    build(config, counter, FakeChatClient())

    assert len(read_lines(train)) > 0


def test_a_finished_build_adds_nothing_when_run_again(
    tmp_path: Path, prompts_dir: Path, counter: TokenCounter
):
    config = make_config(tmp_path, prompts_dir)
    build(config, counter, FakeChatClient())
    before = (config.out_dir / "train.jsonl").read_text()
    client = FakeChatClient()

    build(config, counter, client)

    assert (config.out_dir / "train.jsonl").read_text() == before
    assert client.roles_called == []
