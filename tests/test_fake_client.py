import json
from pathlib import Path

import pytest
from fakes import FakeChatClient, thinking_text

from factory_model_finetune.errors import LmStudioError
from factory_model_finetune.schemas import ROLE_SCHEMAS, valid_reply


def test_fake_client_returns_fixtures(prompts_dir: Path):
    client = FakeChatClient()

    completion = client.complete(
        "spec", "s", "u", thinking=True, schema=None, max_tokens=10, temperature=0
    )

    assert valid_reply(completion.content, ROLE_SCHEMAS["spec"]) is not None
    assert completion.reasoning == thinking_text(1)
    assert client.roles_called == ["spec"]


def test_fake_client_never_repeats_a_reply():
    client = FakeChatClient()
    replies = {
        client.complete(
            "spec", "s", "u", thinking=True, schema=None, max_tokens=10, temperature=0
        ).content
        for _ in range(5)
    }

    assert len(replies) == 5


def test_fake_client_simulates_a_timeout():
    client = FakeChatClient(failing_roles={"plan"})

    with pytest.raises(LmStudioError, match="plan"):
        client.complete("plan", "s", "u", thinking=True, schema=None, max_tokens=1, temperature=0)


def test_fake_client_simulates_a_schema_mismatch():
    client = FakeChatClient(invalid_roles={"spec"})
    completion = client.complete(
        "spec", "s", "u", thinking=True, schema=None, max_tokens=1, temperature=0
    )

    with pytest.raises(json.JSONDecodeError):
        json.loads(completion.content)
    assert valid_reply(completion.content, ROLE_SCHEMAS["spec"]) is None
