from pathlib import Path

import pytest

from factory_model_finetune.errors import ResumeError
from factory_model_finetune.resume import ResumeState, load_state, save_state


def test_resume_loads_and_saves(tmp_path: Path):
    path = tmp_path / "state.json"
    state = ResumeState(
        total_tokens=1234,
        examples_written=7,
        last_job_id="job-9",
        eval_examples_written=1,
        train_examples_by_role={"spec": 5, "plan": 2},
        eval_examples_by_role={"spec": 1},
        tokens_by_role={"spec": 1000, "plan": 234},
        tokens_by_source={"harvest": 1234},
    )

    save_state(path, state)

    assert load_state(path) == state


def test_resume_starts_fresh_without_a_file(tmp_path: Path):
    assert load_state(tmp_path / "state.json") == ResumeState()


def test_resume_save_leaves_no_temporary_file(tmp_path: Path):
    save_state(tmp_path / "state.json", ResumeState())

    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


@pytest.mark.parametrize("content", ["{not json", '{"unknown_field": 1}'])
def test_resume_rejects_unreadable_file(tmp_path: Path, content: str):
    path = tmp_path / "state.json"
    path.write_text(content)

    with pytest.raises(ResumeError, match=r"state\.json"):
        load_state(path)
