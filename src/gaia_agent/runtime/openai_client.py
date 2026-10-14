"""Send chat requests and translate endpoint responses into harness records.

The harness executes tools; the logger consumes the returned usage and metrics.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from urllib.parse import urlparse

import httpx

from src.gaia_agent.runtime.contracts import (
    ChatMessage,
    ContextInfo,
    ModelRequestMetrics,
    ModelTurn,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)


_TOOL_CALL_MARKERS = ("<tool_call>", "</tool_call>")

_VLLM_METRIC_FIELDS = (
    ("time_to_first_token_ms", "time_to_first_token_seconds", 1_000),
    ("generation_time_ms", "generation_seconds", 1_000),
    ("queue_time_ms", "queue_seconds", 1_000),
    ("mean_itl_ms", "mean_inter_token_latency_seconds", 1_000),
    ("tokens_per_second", "tokens_per_second", 1),
)


class OpenAIClientError(RuntimeError):
    """Indicate a transport or response-contract failure."""


class OpenAIChatClient:
    """Call an OpenAI-compatible non-streaming chat-completions endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
    ) -> None:
        parsed_url = urlparse(base_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError("base URL must be an absolute HTTP(S) URL")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if api_key is not None and not isinstance(api_key, str):
            raise TypeError("API key must be a string or None")

        base_path = parsed_url.path.rstrip("/")
        self._url = parsed_url._replace(
            path=f"{base_path}/chat/completions"
        ).geturl()
        self._tokenize_url = parsed_url._replace(
            path=f"{base_path.removesuffix('/v1')}/tokenize"
        ).geturl()
        self._model = model
        self._api_key = api_key

    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition],
        max_completion_tokens: int | None = None,
        enable_thinking: bool | None = None,
    ) -> ModelTurn:
        """Request the next assistant answer or tool calls for this conversation."""
        if max_completion_tokens is not None and (
            type(max_completion_tokens) is not int or max_completion_tokens <= 0
        ):
            raise ValueError("max completion tokens must be a positive integer")
        payload = self._chat_payload(messages, tools, enable_thinking)
        if max_completion_tokens is not None:
            payload["max_completion_tokens"] = max_completion_tokens
        document = await self._post_json(
            self._url, payload, operation="chat completion", generation=True
        )
        turn = self._parse_turn(document)
        if (
            max_completion_tokens is not None
            and turn.usage is not None
            and turn.usage.completion_tokens > max_completion_tokens
        ):
            raise OpenAIClientError("chat completion exceeded its output token limit")
        return turn

    async def context_info(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition],
        enable_thinking: bool | None = None,
    ) -> ContextInfo:
        """Count the deployed chat template and retrieve its actual context limit."""
        payload = self._chat_payload(messages, tools, enable_thinking)
        payload["add_generation_prompt"] = True
        payload["add_special_tokens"] = False
        document = await self._post_json(
            self._tokenize_url, payload, operation="tokenization"
        )
        return _parse_context_info(document)

    async def count_text_tokens(self, text: str) -> int:
        """Count standalone text with the deployed tokenizer, without BOS/EOS."""
        if not isinstance(text, str):
            raise TypeError("tokenization text must be a string")
        document = await self._post_json(
            self._tokenize_url,
            {"model": self._model, "prompt": text, "add_special_tokens": False},
            operation="tokenization",
        )
        return _parse_context_info(document).prompt_tokens

    def _chat_payload(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition],
        enable_thinking: bool | None,
    ) -> dict[str, object]:
        """Share transcript serialization between generation and token counting."""
        if enable_thinking is not None and type(enable_thinking) is not bool:
            raise TypeError("enable thinking must be a boolean or None")
        payload: dict[str, object] = {
            "model": self._model,
            "messages": [self._message_payload(message) for message in messages],
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                        "strict": tool.strict,
                    },
                }
                for tool in tools
            ]
        if enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        return payload

    async def _post_json(
        self,
        url: str,
        payload: dict[str, object],
        *,
        operation: str,
        generation: bool = False,
    ) -> object:
        """Perform one request without exposing endpoint details in safe errors."""
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            # Each invocation owns its transport so concurrent web requests can
            # use the same model adapter from independent event loops.
            async with httpx.AsyncClient() as client:
                # Generation may wait in the inference server's queue.
                if generation:
                    client.timeout.read = None
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
        except httpx.HTTPError as error:
            raise OpenAIClientError(
                f"{operation} request failed: {type(error).__name__}"
            ) from error

        try:
            return response.json()
        except UnicodeDecodeError as error:
            raise OpenAIClientError(f"{operation} response is not UTF-8") from error
        except json.JSONDecodeError as error:
            raise OpenAIClientError(
                f"{operation} response is not valid JSON"
            ) from error

    def _message_payload(self, message: ChatMessage) -> dict[str, object]:
        """Serialize a transcript message, including prior tool calls and results."""
        payload: dict[str, object] = {
            "role": message.role,
            "content": message.content,
        }
        if message.role == "assistant" and message.tool_calls:
            # Qwen's tool-aware chat template expects assistant content to be a
            # string even when the endpoint omitted it for a tool-only turn.
            payload["content"] = message.content or ""
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments_json,
                    },
                }
                for call in message.tool_calls
            ]
        if (
            message.role == "assistant"
            and message.reasoning_content is not None
            and message.reasoning_field is not None
        ):
            payload[message.reasoning_field] = message.reasoning_content
        if message.role == "tool":
            payload["tool_call_id"] = message.tool_call_id
            payload["name"] = message.name
        return payload

    def _parse_turn(self, document: object) -> ModelTurn:
        """Check the response envelope and build the validated harness records."""
        if not isinstance(document, dict):
            raise OpenAIClientError("chat completion response must be a JSON object")
        choices = document.get("choices")
        if not isinstance(choices, list) or not choices:
            raise OpenAIClientError("chat completion response requires a choice")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise OpenAIClientError("chat completion choice must be an object")
        message = choice.get("message")
        if not isinstance(message, dict):
            raise OpenAIClientError("chat completion choice requires a message")
        if message.get("role") != "assistant":
            raise OpenAIClientError("chat completion message role must be assistant")

        content = message.get("content")
        if isinstance(content, str) and any(
            marker in content for marker in _TOOL_CALL_MARKERS
        ):
            raise OpenAIClientError(
                "assistant content contained an unparsed tool-call envelope"
            )
        raw_calls = message.get("tool_calls", [])
        if not isinstance(raw_calls, list):
            raise OpenAIClientError("assistant tool_calls must be an array")

        reasoning_content = None
        reasoning_field = None
        for field_name in ("reasoning_content", "reasoning"):
            value = message.get(field_name)
            if value is None:
                continue
            if reasoning_content is not None and reasoning_content != value:
                raise OpenAIClientError(
                    "assistant reasoning aliases contain conflicting values"
                )
            if reasoning_field is None:
                reasoning_content, reasoning_field = value, field_name

        try:
            response_id = document.get("id")
            turn = ModelTurn(
                content=content,
                reasoning_content=reasoning_content,
                reasoning_field=reasoning_field,
                tool_calls=tuple(_parse_tool_call(raw_call) for raw_call in raw_calls),
                finish_reason=choice.get("finish_reason"),
                response_id=(
                    response_id
                    if isinstance(response_id, str) and response_id.strip() else None
                ),
                usage=_parse_usage(document.get("usage")),
                provider_metrics=_parse_vllm_metrics(document.get("metrics")),
            )
        except (TypeError, ValueError) as error:
            raise OpenAIClientError(f"invalid assistant turn: {error}") from error
        return turn


# ---------------------------------------------------------------------------
# Response field parsing
# ---------------------------------------------------------------------------


def _parse_context_info(document: object) -> ContextInfo:
    """Require exact counts and a usable model limit from the tokenizer endpoint."""
    if not isinstance(document, dict):
        raise OpenAIClientError("tokenization response must be a JSON object")
    count = document.get("count")
    limit = document.get("max_model_len")
    if type(count) is not int or count < 0:
        raise OpenAIClientError("tokenization count must be a non-negative integer")
    if type(limit) is not int or limit <= 0:
        raise OpenAIClientError("tokenization context limit must be a positive integer")
    tokens = document.get("tokens")
    if (
        not isinstance(tokens, list)
        or len(tokens) != count
        or any(type(token) is not int or token < 0 for token in tokens)
    ):
        raise OpenAIClientError("tokenization tokens must match the reported count")
    # Oversized prompts still need their exact count so the runtime can split
    # them into bounded compaction requests rather than truncate their content.
    return ContextInfo(prompt_tokens=count, context_limit_tokens=limit)


def _parse_tool_call(raw_call: object) -> ToolCall:
    """Decode the tool-call envelope and its JSON argument object."""
    if not isinstance(raw_call, dict):
        raise OpenAIClientError("tool call must be an object")
    if raw_call.get("type") != "function":
        raise OpenAIClientError("only function tool calls are supported")
    function = raw_call.get("function")
    if not isinstance(function, dict):
        raise OpenAIClientError("tool call requires a function object")
    raw_arguments = function.get("arguments")
    if not isinstance(raw_arguments, str):
        raise OpenAIClientError("tool call arguments must be a JSON string")
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as error:
        raise OpenAIClientError("tool call arguments are not valid JSON") from error
    if not isinstance(arguments, dict):
        raise OpenAIClientError("tool call arguments must be a JSON object")
    purpose = arguments.get("purpose")
    if not isinstance(purpose, str) or not purpose.strip() or len(purpose) > 300:
        raise OpenAIClientError(
            "tool call purpose must be a non-empty string of at most 300 characters"
        )
    executable_arguments = {
        name: value for name, value in arguments.items() if name != "purpose"
    }
    return ToolCall(
        id=raw_call.get("id"),
        name=function.get("name"),
        arguments=executable_arguments,
        purpose=purpose,
        emitted_arguments=arguments,
    )


def _parse_usage(raw_usage: object) -> TokenUsage | None:
    """Keep usable token accounting without making it a response requirement."""
    if not isinstance(raw_usage, dict):
        return None
    counts: dict[str, int] = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = raw_usage.get(name)
        if type(value) is not int or value < 0:
            return None
        counts[name] = value
    if counts["total_tokens"] != counts["prompt_tokens"] + counts["completion_tokens"]:
        return None
    for group, key, field, parent in (
        ("prompt_tokens_details", "cached_tokens", "cached_prompt_tokens", "prompt_tokens"),
        ("completion_tokens_details", "reasoning_tokens", "reasoning_tokens", "completion_tokens"),
    ):
        details = raw_usage.get(group)
        value = details.get(key) if isinstance(details, dict) else None
        if type(value) is int and 0 <= value <= counts[parent]:
            counts[field] = value
    return TokenUsage(**counts)


def _parse_vllm_metrics(raw_metrics: object) -> ModelRequestMetrics | None:
    """Convert usable vLLM metrics to seconds, omitting malformed fields."""
    if not isinstance(raw_metrics, dict):
        return None
    values: dict[str, float] = {}
    for response_field, runtime_field, divisor in _VLLM_METRIC_FIELDS:
        value = raw_metrics.get(response_field)
        if type(value) not in (int, float):
            continue
        try:
            value = value / divisor
        except OverflowError:
            continue
        if math.isfinite(value) and value >= 0:
            values[runtime_field] = value
    return ModelRequestMetrics(provider="vllm", **values) if values else None
