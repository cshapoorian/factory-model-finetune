from pathlib import Path

import pytest

from factory_model_finetune.errors import TokenizerError
from factory_model_finetune.token_counter import MESSAGE_OVERHEAD_TOKENS, TokenCounter


def test_token_counter_validates_path_missing_file(tmp_path: Path):
    missing = tmp_path / "nope" / "tokenizer.json"

    with pytest.raises(TokenizerError, match="not found") as error:
        TokenCounter(missing)

    assert str(missing) in str(error.value)


def test_token_counter_validates_path_corrupted_file(tmp_path: Path):
    corrupted = tmp_path / "tokenizer.json"
    corrupted.write_text("this is not a tokenizer")

    with pytest.raises(TokenizerError, match="corrupted") as error:
        TokenCounter(corrupted)

    assert str(corrupted) in str(error.value)


def test_token_counter_counts_tokens(counter: TokenCounter):
    assert counter.count("one two three") == 3
    assert counter.count("") == 0


def test_token_counter_counts_message_overhead(counter: TokenCounter):
    messages = [{"role": "user", "content": "one two"}, {"role": "assistant", "content": "three"}]

    assert counter.count_messages(messages) == 3 + 2 * MESSAGE_OVERHEAD_TOKENS
