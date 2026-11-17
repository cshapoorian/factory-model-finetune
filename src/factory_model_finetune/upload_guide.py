"""Writes the instructions for uploading the dataset; this tool never runs `firectl` itself."""

from pathlib import Path

# The Fireworks id of Qwen3.6-35B-A3B is not known to this tool, so the guide leaves it to fill in.
BASE_MODEL_PLACEHOLDER = "accounts/fireworks/models/<qwen3.6-35b-a3b-model-id>"


def render_upload_guide(
    out_dir: Path, train_tokens: int, epochs: int, price_per_million: float
) -> str:
    """Return the text of `UPLOAD.md` with the exact commands to run.

    `out_dir` is shown as given, so the commands work from where the build was started.
    """
    billed_tokens = train_tokens * epochs
    cost = billed_tokens / 1_000_000 * price_per_million
    return f"""# Uploading the dataset to Fireworks

Nothing has been uploaded and `firectl` has not been run. Run these commands yourself.

Placeholders for you to fill in:
- `<account-id>`: your Fireworks account id
- `<dataset-name>`: a name you choose for the dataset
- `<output-model-name>`: a name you choose for the fine-tuned model
- `<qwen3.6-35b-a3b-model-id>`: the base model's id in the Fireworks model library

## 1. Sign in

```sh
firectl signin
```

## 2. Create the datasets

```sh
firectl dataset create <dataset-name> {out_dir / "train.jsonl"}
firectl dataset create <dataset-name>-eval {out_dir / "eval.jsonl"}
```

## 3. Start the fine-tuning job

```sh
firectl create sftj \\
  --base-model {BASE_MODEL_PLACEHOLDER} \\
  --dataset accounts/<account-id>/datasets/<dataset-name> \\
  --evaluation-dataset accounts/<account-id>/datasets/<dataset-name>-eval \\
  --output-model <output-model-name> \\
  --epochs {epochs}
```

Run `firectl create sftj --help` first to confirm the flag names for your `firectl` version.

## Cost

`train.jsonl` holds {train_tokens:,} tokens. At {epochs} epoch(s) that is {billed_tokens:,} billed
tokens, about ${cost:.2f} at ${price_per_million:.2f} per 1M training tokens. `eval.jsonl` is
held out and is not part of that figure.
"""
