"""Errors this tool raises on purpose; the CLI turns them into a message and a non-zero exit."""


class FinetuneError(Exception):
    """Base class for every deliberate error in this package."""


class ConfigError(FinetuneError):
    """A command-line value or input directory is unusable."""


class TokenizerError(FinetuneError):
    """The tokenizer file is missing or corrupted."""


class ResumeError(FinetuneError):
    """The resume file exists but cannot be read."""


class LmStudioError(FinetuneError):
    """LM Studio is unreachable, timed out, or answered with an error."""
