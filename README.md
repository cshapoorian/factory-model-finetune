# factory-model-finetune

Builds a Fireworks fine-tuning dataset (`train.jsonl`, `eval.jsonl`) for the software factory's
local Qwen3.6-35B-A3B model. Every example, and every reasoning trace, comes from the factory's own
logs or from that same local model running in LM Studio; no Claude output is used.

## Usage

```sh
uv run python -m factory_model_finetune build \
  --factory-runs "~/repos/software factory/runs" --out data/
```

Options: `--budget-usd` (15), `--price-per-million` (2.10), `--epochs` (1), `--role-mix`
(like `spec=30,plan=20`), `--eval-fraction` (0.03), `--prompts-dir`, `--tokenizer`, `--seed`,
`--base-url` (`http://127.0.0.1:1234/v1`) and `--model` (`qwen/qwen3.6-35b-a3b`).

The target size is `budget / price * 1M / epochs` tokens, counted with the model's tokenizer.
The build is resumable: stop it any time (Ctrl-C) and run the same command again; progress is kept
in `data/state.json`. When it finishes it prints the token count, estimated cost and the split by
role and source, and writes `data/UPLOAD.md` with the `firectl` commands for you to run. The tool
never uploads anything.

## Development

```sh
uv sync
uv run pytest
uv run ruff check . && uv run pyright
```

## Configuration

This project reads no environment variables. Everything is set with the command-line options above.
