"""Format model calls into Fireworks fine-tuning examples using the OpenAI chat schema.

This module handles the transformation of raw model outputs (including optional
reasoning traces) into structured message lists suitable for training. It enforces
constraints on reasoning roles and normalizes the output format for the Qwen3.6
chat template.
"""

import json
from dataclasses import dataclass

from factory_model_finetune.roles import REASONING_ROLES

# What the Qwen3.6 chat template puts at the start of a non-thinking assistant turn.
EMPTY_THINK_BLOCK = "<think>\n\n</think>\n\n"
THINK_END = "</think>\n\n"


@dataclass(frozen=True)
class Candidate:
    """Represents a single model interaction ready for fine-tuning.

    Attributes:
        role: The model's role identifier (e.g., 'reasoning' or 'text').
        source: Origin identifier for the candidate.
        system: The system prompt provided to the model.
        user: The user prompt provided to the model.
        reasoning: The model's internal reasoning trace. Empty for non-reasoning roles.
        reply: The final response. Normalised JSON for reasoning roles, plain text otherwise.
    """

    role: str
    source: str
    system: str
    user: str
    reasoning: str
    reply: str


@dataclass(frozen=True)
class Example:
    """A complete training example including token cost estimation.

    Attributes:
        role: The model's role identifier.
        source: Origin identifier for the example.
        messages: List of message dicts in OpenAI chat schema format.
        tokens: Estimated token count for this example.
    """

    role: str
    source: str
    messages: list[dict[str, str]]
    tokens: int


def assistant_content(role: str, reasoning: str, reply: str) -> str:
    """Construct the assistant message content based on role type.

    For reasoning roles, wraps the reasoning trace in think tags. For non-reasoning
    roles, inserts an empty think block to maintain template consistency.

    Args:
        role: The model's role identifier.
        reasoning: The reasoning trace string. Must be non-empty for reasoning roles.
        reply: The final response text.

    Returns:
        The formatted assistant content string.

    Raises:
        ValueError: If a reasoning role is provided without a reasoning trace.
    """
    # Non-reasoning roles require an empty think block to match the expected
    # tokenization and structure of reasoning-capable models.
    if role not in REASONING_ROLES:
        return f"{EMPTY_THINK_BLOCK}{reply}"
    if not reasoning.strip():
        raise ValueError(f"role {role!r} is a reasoning role but the example has no reasoning")
    # Strip reasoning to remove potential leading/trailing whitespace artifacts
    # from model generation, ensuring clean tag boundaries.
    return f"<think>{reasoning.strip()}</think>\n\n{reply}"


def to_messages(candidate: Candidate) -> list[dict[str, str]]:
    """Convert a Candidate into a list of OpenAI-style chat messages.

    Constructs the standard system, user, and assistant message sequence required
    for fine-tuning. The assistant message is generated via `assistant_content`.

    Args:
        candidate: The candidate data to format.

    Returns:
        A list of message dictionaries ready for JSON serialization.
    """
    # The order of messages is critical for the chat template to correctly
    # associate the system prompt with the subsequent conversation.
    return [
        {"role": "system", "content": candidate.system},
        {"role": "user", "content": candidate.user},
        {
            "role": "assistant",
            "content": assistant_content(candidate.role, candidate.reasoning, candidate.reply),
        },
    ]


def to_jsonl_line(messages: list[dict[str, str]]) -> str:
    """Serialize a list of messages into a single JSONL line.

    Uses `ensure_ascii=False` to preserve unicode characters in the content,
    which is critical for multilingual training data.

    Args:
        messages: The list of message dictionaries.

    Returns:
        A JSON string representing the line, without a trailing newline.
    """
    return json.dumps({"messages": messages}, ensure_ascii=False)


def final_answer(assistant_text: str) -> str:
    """Extract the final response from an assistant turn, stripping the think block.

    Parses the assistant text to find the `THINK_END` separator. If found, returns
    the content following it. If not found (e.g., in non-reasoning outputs or
    malformed data), returns the original text unchanged.

    Args:
        assistant_text: The full assistant message content.

    Returns:
        The response text after the think block, or the original text if no block exists.
    """
    # Use partition to efficiently split on the first occurrence of the separator.
    # This avoids regex overhead and handles cases where the separator might appear
    # multiple times in the reasoning trace (though unlikely with proper formatting).
    _, separator, answer = assistant_text.partition(THINK_END)
    return answer if separator else assistant_text
