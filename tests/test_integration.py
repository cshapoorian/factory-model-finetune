import json
from pathlib import Path

import pytest
from fakes import FakeChatClient

from factory_model_finetune import cli
from factory_model_finetune.data_formatter import EMPTY_THINK_BLOCK


@pytest.fixture
def run_cli(
    tmp_path: Path,
    tokenizer_path: Path,
    prompts_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """Run the real CLI with the fake model in place of LM Studio."""
    monkeypatch.setattr(cli, "LmStudioClient", lambda base_url, model: FakeChatClient())

    def run(*extra: str, tokenizer: Path | None = None) -> tuple[int, str, str]:
        code = cli.main(
            [
                "build",
                "--factory-runs",
                str(tmp_path / "runs"),
                "--out",
                str(tmp_path / "data"),
                "--prompts-dir",
                str(prompts_dir),
                "--tokenizer",
                str(tokenizer or tokenizer_path),
                "--budget-usd",
                "0.01",
                "--eval-fraction",
                "0.2",
                *extra,
            ]
        )
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    return run


def test_end_to_end_build_no_network(run_cli, tmp_path: Path):
    code, out, _ = run_cli()

    data = tmp_path / "data"
    assert code == 0
    assert {p.name for p in data.iterdir()} >= {"train.jsonl", "eval.jsonl", "UPLOAD.md"}
    assert "Estimated cost" in out
    assert "By role" in out


def test_all_criteria_covered(run_cli, tmp_path: Path):
    run_cli("--epochs", "2")

    data = tmp_path / "data"
    examples = [
        json.loads(line)["messages"]
        for name in ("train.jsonl", "eval.jsonl")
        for line in (data / name).read_text().splitlines()
    ]
    assert examples
    for system, user, assistant in examples:
        assert (system["role"], user["role"], assistant["role"]) == ("system", "user", "assistant")
        content = assistant["content"]
        if content.startswith(EMPTY_THINK_BLOCK):
            continue
        reasoning, _, answer = content.removeprefix("<think>").partition("</think>\n\n")
        assert reasoning.strip()
        assert isinstance(json.loads(answer), dict)
    guide = (data / "UPLOAD.md").read_text()
    assert "firectl signin" in guide
    assert "--epochs 2" in guide


def test_reports_a_missing_tokenizer_by_path(run_cli, tmp_path: Path):
    missing = tmp_path / "nope" / "tokenizer.json"

    code, _, err = run_cli(tokenizer=missing)

    assert code == 1
    assert str(missing) in err
