import socket
from pathlib import Path

import pytest
from fakes import write_prompts, write_tokenizer

from factory_model_finetune.token_counter import TokenCounter


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if any test tries to open a real connection (AC7)."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise RuntimeError("tests must not use the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)


@pytest.fixture
def tokenizer_path(tmp_path: Path) -> Path:
    return write_tokenizer(tmp_path / "tokenizer.json")


@pytest.fixture
def counter(tokenizer_path: Path) -> TokenCounter:
    return TokenCounter(tokenizer_path)


@pytest.fixture
def prompts_dir(tmp_path: Path) -> Path:
    return write_prompts(tmp_path / "prompts")
