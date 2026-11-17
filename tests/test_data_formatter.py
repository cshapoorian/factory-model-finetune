import json

import pytest

from factory_model_finetune.data_formatter import (
    EMPTY_THINK_BLOCK,
    Candidate,
    final_answer,
    to_jsonl_line,
    to_messages,
)


def candidate(role: str, reasoning: str = "", reply: str = '{"ok": true}') -> Candidate:
    return Candidate(role, "synthetic", "the system prompt", "the user prompt", reasoning, reply)


@pytest.mark.parametrize("role", ["spec", "plan", "review"])
def test_formatter_reasoning_blocks(role: str):
    messages = to_messages(candidate(role, reasoning="  weighing options  "))

    assert [m["role"] for m in messages] == ["system", "user", "assistant"]
    assert messages[2]["content"] == '<think>weighing options</think>\n\n{"ok": true}'


@pytest.mark.parametrize(
    "role",
    [
        "trivial_edit",
        "docs_pass",
        "evidence",
        "summarize",
        "commit_message",
        "pr_body",
        "detect_profile",
        "memory",
    ],
)
def test_formatter_non_reasoning_empty_think(role: str):
    messages = to_messages(candidate(role, reasoning="ignored"))

    assert messages[2]["content"] == f'{EMPTY_THINK_BLOCK}{{"ok": true}}'
    assert messages[2]["content"].startswith("<think>\n\n</think>\n\n")


def test_formatter_rejects_reasoning_role_without_reasoning():
    with pytest.raises(ValueError, match="spec"):
        to_messages(candidate("spec", reasoning="  "))


def test_jsonl_line_holds_only_messages():
    line = to_jsonl_line(to_messages(candidate("memory")))

    assert list(json.loads(line)) == ["messages"]
    assert "\n" not in line


@pytest.mark.parametrize("role", ["spec", "memory"])
def test_final_answer_strips_the_think_block(role: str):
    content = to_messages(candidate(role, reasoning="hmm", reply="the answer"))[2]["content"]

    assert final_answer(content) == "the answer"
