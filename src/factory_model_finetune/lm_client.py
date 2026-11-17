"""Client for LM Studio's OpenAI-compatible chat API, the same one the factory talks to."""

import logging
from dataclasses import dataclass
from typing import Protocol

import httpx

from factory_model_finetune.errors import LmStudioError
from factory_model_finetune.schemas import Schema

logger = logging.getLogger(__name__)

# One reasoning call can run for many minutes on a local model, so the timeout is generous.
DEFAULT_TIMEOUT_SECONDS = 1800


@dataclass(frozen=True)
class Completion:
    """A model's answer: the reply text and, when reasoning was on, the reasoning text."""

    content: str
    reasoning: str


class ChatClient(Protocol):
    """What the generator needs from a model; tests supply a fake."""

    def complete(
        self,
        role: str,
        system: str,
        user: str,
        *,
        thinking: bool,
        schema: Schema | None,
        max_tokens: int,
        temperature: float,
    ) -> Completion:
        """Answer one system and user message pair.

        Raises:
            LmStudioError: the model could not be reached or answered with an error.
        """
        ...


class LmStudioClient:
    """Calls LM Studio, building requests the way the factory does.

    Args:
        base_url: The base URL of the LM Studio server.
        model: The model identifier to use for completions.
        timeout_seconds: Timeout for HTTP requests in seconds. Defaults to 30 minutes
            to accommodate long reasoning traces on local models.
        transport: Optional custom httpx transport for testing or proxying.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._model = model
        self._http = httpx.Client(base_url=base_url, timeout=timeout_seconds, transport=transport)

    def complete(
        self,
        role: str,
        system: str,
        user: str,
        *,
        thinking: bool,
        schema: Schema | None,
        max_tokens: int,
        temperature: float,
    ) -> Completion:
        """Ask the model, retrying once if the server is unreachable, times out or errors (5xx).

        Args:
            role: The role name, used as the JSON schema name if `schema` is provided.
            system: The system prompt.
            user: The user prompt.
            thinking: If True, enables reasoning effort; otherwise sets it to "none".
            schema: Optional JSON schema for structured output. Callers pass None when
                `thinking` is on, because LM Studio would then put the JSON in the reasoning
                channel; the schema is described in the prompt instead.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.

        Returns:
            The model's completion, including content and optional reasoning text.

        Raises:
            LmStudioError: If the model fails to respond after two attempts, or if the response
                format is unexpected.
        """
        body: dict[str, object] = {
            "model": self._model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if not thinking:
            body["reasoning_effort"] = "none"
        if schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": role, "strict": True, "schema": schema},
            }
        failure = ""
        for attempt in (1, 2):
            try:
                response = self._http.post("/chat/completions", json=body)
            except httpx.TransportError as error:
                failure = f"{type(error).__name__}: {error}"
                logger.warning(
                    "LM Studio call for %s failed (attempt %d): %s", role, attempt, failure
                )
                continue
            if response.status_code == 200:
                return _completion_from(response.json(), role)
            failure = f"HTTP {response.status_code}: {response.text[:200]}"
            logger.warning("LM Studio call for %s failed (attempt %d): %s", role, attempt, failure)
            if response.status_code < 500:
                break
        raise LmStudioError(
            f"LM Studio at {self._http.base_url} failed for role {role!r}: {failure}"
        )


def _completion_from(data: dict, role: str) -> Completion:
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as error:
        raise LmStudioError(
            f"unexpected LM Studio response for role {role!r}: {data!r:.200}"
        ) from error
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    return Completion(content=message.get("content") or "", reasoning=reasoning)
