"""Counts tokens with the fine-tuned model's own tokenizer, so the budget matches what is billed."""

from pathlib import Path

from tokenizers import Tokenizer

from factory_model_finetune.errors import TokenizerError

# Fireworks bills the chat-templated text, which wraps each message in role markers and an end
# marker (about 4 tokens). Counting them keeps the estimate from drifting low on short messages.
MESSAGE_OVERHEAD_TOKENS = 4


class TokenCounter:
    """Counts tokens for text and chat messages using a `tokenizer.json` file."""

    def __init__(self, tokenizer_path: Path) -> None:
        """Load the tokenizer.

        Raises:
            TokenizerError: the file is missing or is not a valid tokenizer.
        """
        path = tokenizer_path.expanduser()
        if not path.is_file():
            raise TokenizerError(f"tokenizer file not found at {path}")
        try:
            self._tokenizer = Tokenizer.from_file(str(path))
        except Exception as error:  # `tokenizers` raises a bare Exception for bad files
            raise TokenizerError(f"tokenizer file at {path} is corrupted: {error}") from error

    def count(self, text: str) -> int:
        """Return the number of tokens in `text`."""
        return len(self._tokenizer.encode(text).ids)

    def count_messages(self, messages: list[dict[str, str]]) -> int:
        """Return the tokens a chat example costs: every message's text plus template overhead."""
        return sum(self.count(m["content"]) + MESSAGE_OVERHEAD_TOKENS for m in messages)
