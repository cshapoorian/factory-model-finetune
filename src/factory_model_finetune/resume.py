"""The small JSON file that lets an interrupted build continue where it stopped."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from factory_model_finetune.errors import ResumeError


@dataclass
class ResumeState:
    """Progress of a build. Token and example counts cover `train.jsonl` only (what is billed).

    `last_job_id` is the last factory job whose log was fully harvested; jobs are read in name
    order, so a resumed run skips everything up to and including it.
    """

    total_tokens: int = 0
    examples_written: int = 0
    last_job_id: str | None = None
    eval_examples_written: int = 0
    train_examples_by_role: dict[str, int] = field(default_factory=dict)
    eval_examples_by_role: dict[str, int] = field(default_factory=dict)
    tokens_by_role: dict[str, int] = field(default_factory=dict)
    tokens_by_source: dict[str, int] = field(default_factory=dict)


def load_state(path: Path) -> ResumeState:
    """Return the saved state, or a fresh one when no file exists yet.

    Raises:
        ResumeError: the file exists but is not a valid state file.
    """
    if not path.exists():
        return ResumeState()
    try:
        return ResumeState(**json.loads(path.read_text()))
    except (json.JSONDecodeError, TypeError) as error:
        raise ResumeError(
            f"resume file {path} is unreadable ({error}); delete it to start over"
        ) from error


def save_state(path: Path, state: ResumeState) -> None:
    """Write the state atomically, so an interrupt never leaves a half-written file."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(asdict(state), indent=2))
    temporary.replace(path)
