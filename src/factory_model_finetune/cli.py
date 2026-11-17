"""Command line: `python -m factory_model_finetune build --factory-runs <runs> --out <folder>`."""

import argparse
import logging
import sys
from pathlib import Path

from factory_model_finetune.builder import BuildConfig, DatasetBuilder
from factory_model_finetune.errors import FinetuneError
from factory_model_finetune.lm_client import LmStudioClient
from factory_model_finetune.roles import parse_role_mix
from factory_model_finetune.token_counter import TokenCounter

DEFAULT_TOKENIZER = "~/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-6bit/tokenizer.json"
DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_MODEL = "qwen/qwen3.6-35b-a3b"
INTERRUPTED_EXIT_CODE = 130


def build_parser() -> argparse.ArgumentParser:
    """
    Construct and return the CLI argument parser.

    Defines subcommands (currently only 'build') and their respective
    configuration options for generating synthetic fine-tuning data.
    """
    parser = argparse.ArgumentParser(
        prog="factory_model_finetune",
        description="Build a Fireworks fine-tuning dataset for the factory's local model.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser(
        "build", help="harvest the factory's logs, generate synthetic examples, write the dataset"
    )
    build.add_argument(
        "--factory-runs", type=Path, required=True, help="the factory's runs/ folder"
    )
    build.add_argument("--out", type=Path, required=True, help="folder for train.jsonl and friends")
    build.add_argument(
        "--prompts-dir",
        type=Path,
        help="the factory's prompts folder (default: <factory-runs>/../factory/prompts)",
    )
    build.add_argument("--tokenizer", type=Path, default=Path(DEFAULT_TOKENIZER))
    build.add_argument("--budget-usd", type=float, default=15.0)
    build.add_argument("--price-per-million", type=float, default=2.10)
    build.add_argument("--epochs", type=int, default=1)
    build.add_argument(
        "--role-mix", default="", help="token-share weights to override, like spec=30,plan=20"
    )
    build.add_argument("--eval-fraction", type=float, default=0.03)
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--base-url", default=DEFAULT_BASE_URL)
    build.add_argument("--model", default=DEFAULT_MODEL)
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Execute the CLI workflow and return the exit code.

    Parses arguments, initializes the dataset builder with the specified
    configuration and tokenizer, and triggers the build process against
    the local LM Studio instance.

    Returns:
        0 on success, 1 on application error, 130 on keyboard interrupt.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(levelname)s %(message)s"
    )
    try:
        factory_runs = args.factory_runs.expanduser()
        config = BuildConfig(
            factory_runs=factory_runs,
            out_dir=args.out,
            prompts_dir=(
                args.prompts_dir or factory_runs.parent / "factory" / "prompts"
            ).expanduser(),
            budget_usd=args.budget_usd,
            price_per_million=args.price_per_million,
            epochs=args.epochs,
            role_mix=parse_role_mix(args.role_mix),
            eval_fraction=args.eval_fraction,
            seed=args.seed,
        )
        builder = DatasetBuilder(config, TokenCounter(args.tokenizer))
        report = builder.build(LmStudioClient(args.base_url, args.model))
    except FinetuneError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(
            "interrupted; progress is saved. Run the same command again to resume.", file=sys.stderr
        )
        return INTERRUPTED_EXIT_CODE
    print(report.format())
    return 0


def run() -> None:
    """
    Entry point for console scripts (e.g., `factory_model_finetune`).

    Wraps `main()` in a SystemExit to ensure the correct exit code is
    propagated to the shell.
    """
    raise SystemExit(main())
