import json
from typing import Any

import pytest
from fakes import make_plan, make_review, make_spec

from factory_model_finetune.data_formatter import Candidate
from factory_model_finetune.factory_text import spec_markdown
from factory_model_finetune.quality_filter import (
    CONSISTENCY,
    DUPLICATE,
    NON_TRIVIAL,
    DuplicateIndex,
    failed_filter,
)

THINKING = "Reasoning about the problem. " * 20


def candidate(role: str, user: str, value: dict[str, Any] | str, reasoning: str = "") -> Candidate:
    reply = value if isinstance(value, str) else json.dumps(value)
    return Candidate(role, "synthetic", "system", user, reasoning, reply)


def plan_candidate(covers: list[str]) -> Candidate:
    user = spec_markdown(make_spec(1))
    return candidate("plan", user, make_plan(1, covers), THINKING)


def test_accepts_a_good_plan():
    good = plan_candidate(["AC1", "AC2", "AC3", "EC1", "EC2"])

    assert failed_filter(good, DuplicateIndex()) is None


def test_rejects_a_plan_that_leaves_a_criterion_uncovered():
    assert failed_filter(plan_candidate(["AC1", "AC2", "EC1", "EC2"]), DuplicateIndex()) == (
        CONSISTENCY
    )


def test_rejects_a_plan_covering_an_unknown_id():
    covers = ["AC1", "AC2", "AC3", "EC1", "EC2", "AC9"]

    assert failed_filter(plan_candidate(covers), DuplicateIndex()) == CONSISTENCY


def test_rejects_a_spec_with_duplicate_ids():
    spec = make_spec(1)
    spec["acceptance_criteria"][1]["id"] = "AC1"

    assert failed_filter(candidate("spec", "u", spec, THINKING), DuplicateIndex()) == CONSISTENCY


def test_rejects_a_spec_without_enough_substance():
    spec = make_spec(1, criteria=1)

    assert failed_filter(candidate("spec", "u", spec, THINKING), DuplicateIndex()) == NON_TRIVIAL


def test_rejects_a_reasoning_role_with_thin_reasoning():
    assert (
        failed_filter(candidate("spec", "u", make_spec(1), "hm"), DuplicateIndex()) == NON_TRIVIAL
    )


def test_rejects_a_pass_verdict_with_a_major_finding():
    review = make_review(1, passes=False)
    review["verdict"] = "pass"
    diff = "diff --git a/src/app/service.py b/src/app/service.py"

    assert failed_filter(candidate("review", diff, review, THINKING), DuplicateIndex()) == (
        CONSISTENCY
    )


@pytest.mark.parametrize(
    "message",
    [
        "Added a thing to the code base today, finally",
        "feat: ok\nno blank second line follows this one",
    ],
)
def test_rejects_a_malformed_commit_message(message: str):
    assert failed_filter(candidate("commit_message", "u", message), DuplicateIndex()) == (
        CONSISTENCY
    )


def test_rejects_a_reply_that_is_a_near_copy_of_an_earlier_one():
    index = DuplicateIndex()
    original = " ".join(f"w{i}" for i in range(40))
    index.add("system", "first prompt", original)
    near_copy = candidate("summarize", "second prompt", original + " extra")

    assert failed_filter(near_copy, index) == DUPLICATE


def test_does_not_flag_a_different_reply_as_duplicate():
    index = DuplicateIndex()
    index.add("system", "p1", " ".join(f"a{i}" for i in range(40)))

    assert not index.is_duplicate("system", "p2", " ".join(f"b{i}" for i in range(40)))


def test_flags_a_repeated_prompt_even_when_the_reply_differs():
    index = DuplicateIndex()
    index.add("system", "same prompt", "one reply")

    assert index.is_duplicate("system", "same prompt", "something else entirely")


def test_does_not_compare_replies_across_roles():
    index = DuplicateIndex()
    reply = " ".join(f"w{i}" for i in range(40))
    index.add("summarize system", "p1", reply)

    assert not index.is_duplicate("pr system", "p2", reply)


def test_lets_short_replies_repeat():
    index = DuplicateIndex()
    index.add("system", "p1", '{"profile": "python"}')

    assert not index.is_duplicate("system", "p2", '{"profile": "python"}')
