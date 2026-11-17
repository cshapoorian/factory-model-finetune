import logging
import random
from pathlib import Path

import httpx
import pytest
from fakes import FakeChatClient

from factory_model_finetune.data_formatter import Candidate
from factory_model_finetune.errors import LmStudioError
from factory_model_finetune.lm_client import Completion, LmStudioClient
from factory_model_finetune.progress_reporter import ProgressReporter
from factory_model_finetune.quality_filter import DuplicateIndex, failed_filter
from factory_model_finetune.roles import ALL_ROLES, PROFILES, REASONING_ROLES
from factory_model_finetune.synthetic_generator import (
    MAX_CONSECUTIVE_LM_ERRORS,
    SyntheticGenerator,
)


class Collector:
    """Stands in for the builder: applies the real filters and keeps what passes."""

    def __init__(self, reporter: ProgressReporter) -> None:
        self.accepted: list[Candidate] = []
        self._index = DuplicateIndex()
        self._reporter = reporter

    def __call__(self, candidate: Candidate) -> bool:
        if rejected_by := failed_filter(candidate, self._index):
            self._reporter.rejected(rejected_by)
            return False
        self._index.add(candidate.system, candidate.user, candidate.reply)
        self.accepted.append(candidate)
        return True


def make_generator(
    client: FakeChatClient, prompts_dir: Path
) -> tuple[SyntheticGenerator, Collector]:
    reporter = ProgressReporter(target_tokens=1000)
    collector = Collector(reporter)
    return SyntheticGenerator(client, prompts_dir, reporter, collector, random.Random(0)), collector


def test_generates_across_profiles(prompts_dir: Path):
    generator, collector = make_generator(FakeChatClient(), prompts_dir)

    for profile in PROFILES:
        generator.run_scenario(profile, {"spec"})

    profile_texts = {c.user.split("# Profile\n")[1].split("\n\n")[0] for c in collector.accepted}
    assert len(collector.accepted) == len(PROFILES)
    assert len(profile_texts) == len(PROFILES)


def test_produces_every_role(prompts_dir: Path):
    generator, collector = make_generator(FakeChatClient(), prompts_dir)

    generator.run_scenario("python", set(ALL_ROLES))

    assert {c.role for c in collector.accepted} == set(ALL_ROLES)


def test_reasoning_roles_keep_their_reasoning(prompts_dir: Path):
    generator, collector = make_generator(FakeChatClient(), prompts_dir)

    generator.run_scenario("python", set(ALL_ROLES))

    for candidate in collector.accepted:
        assert bool(candidate.reasoning) == (candidate.role in REASONING_ROLES)


def test_skips_the_example_when_lm_studio_fails(
    prompts_dir: Path, caplog: pytest.LogCaptureFixture
):
    generator, collector = make_generator(FakeChatClient(failing_roles={"scenario"}), prompts_dir)

    with caplog.at_level(logging.ERROR):
        generator.run_scenario("python", {"spec"})

    assert collector.accepted == []
    assert "skipping this example" in caplog.text


def test_stops_after_too_many_consecutive_lm_errors(prompts_dir: Path):
    generator, _ = make_generator(FakeChatClient(failing_roles={"scenario"}), prompts_dir)

    with pytest.raises(LmStudioError, match="in a row"):
        for _ in range(MAX_CONSECUTIVE_LM_ERRORS):
            generator.run_scenario("python", {"spec"})


def test_drops_a_reply_that_does_not_match_its_schema(prompts_dir: Path):
    generator, collector = make_generator(FakeChatClient(invalid_roles={"spec"}), prompts_dir)

    generator.run_scenario("python", {"spec"})

    assert collector.accepted == []


# ---- the real client's single retry (EC2) ---------------------------------------


def lm_studio_client(handler: httpx.MockTransport) -> LmStudioClient:
    return LmStudioClient("http://lm.test/v1", "m", transport=handler)


def ask(client: LmStudioClient, *, thinking: bool = False) -> Completion:
    return client.complete(
        "summarize", "s", "u", thinking=thinking, schema=None, max_tokens=10, temperature=0
    )


def test_retries_on_timeout():
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            raise httpx.ReadTimeout("timed out")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    completion = ask(lm_studio_client(httpx.MockTransport(handler)))

    assert completion.content == "ok"
    assert len(attempts) == 2


def test_gives_up_after_one_retry():
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        raise httpx.ConnectError("unreachable")

    with pytest.raises(LmStudioError, match="unreachable"):
        ask(lm_studio_client(httpx.MockTransport(handler)))
    assert len(attempts) == 2


def test_does_not_retry_a_client_error():
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(400, text="bad request")

    with pytest.raises(LmStudioError, match="400"):
        ask(lm_studio_client(httpx.MockTransport(handler)))
    assert len(attempts) == 1


def test_reads_reasoning_from_the_reply():
    body = {"choices": [{"message": {"content": "answer", "reasoning_content": "because"}}]}
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=body))

    completion = ask(lm_studio_client(transport), thinking=True)

    assert (completion.content, completion.reasoning) == ("answer", "because")
