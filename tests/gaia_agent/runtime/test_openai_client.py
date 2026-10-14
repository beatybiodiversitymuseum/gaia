"""Protocol and asynchronous transport tests for the model adapter."""

from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from functools import partial
from unittest.mock import Mock, patch

import httpx

import pytest

from src.gaia_agent.runtime.contracts import (
    ChatMessage,
    ContextInfo,
    ModelRequestMetrics,
    ModelTurn,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)
from src.gaia_agent.runtime.openai_client import (
    OpenAIChatClient,
    OpenAIClientError,
)


pytestmark = pytest.mark.anyio


def json_response(document: object) -> httpx.Response:
    return httpx.Response(
        200, content=json.dumps(document, ensure_ascii=False).encode("utf-8")
    )


@contextmanager
def mock_endpoint(*responses: httpx.Response):
    handler = Mock(side_effect=responses)
    factory = partial(httpx.AsyncClient, transport=httpx.MockTransport(handler))
    with patch("src.gaia_agent.runtime.openai_client.httpx.AsyncClient", factory):
        yield handler


async def _complete_response(document: object) -> ModelTurn:
    client = OpenAIChatClient(base_url="http://localhost/v1", model="test")
    with mock_endpoint(json_response(document)):
        return await client.complete(
            messages=(ChatMessage(role="user", content="Test request."),),
            tools=(),
        )


async def test_serializes_transcript_and_parses_parallel_tool_calls() -> None:
    response = json_response(
        {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "lookup",
                                    "arguments": '{"purpose":"Check the first value.","value":2}',
                                },
                            },
                            {
                                "id": "call-2",
                                "type": "function",
                                "function": {
                                    "name": "lookup",
                                    "arguments": '{"value":3,"purpose":"Check the second value."}',
                                },
                            },
                        ],
                    },
                }
            ]
        }
    )
    client = OpenAIChatClient(
        base_url="http://127.0.0.1:8000/v1/",
        model="local-model",
        api_key="test-key",
    )
    previous_call = ToolCall("old-call", "lookup", {"value": 1})
    messages = (
        ChatMessage(role="user", content="Look these up."),
        ChatMessage(role="assistant", content=None, tool_calls=(previous_call,)),
        ChatMessage(
            role="tool",
            content='{"ok":true,"result":1}',
            tool_call_id="old-call",
            name="lookup",
        ),
    )
    definition = ToolDefinition(
        "lookup",
        "Look up one value.",
        {"type": "object", "properties": {"value": {"type": "integer"}}},
        strict=True,
    )

    with mock_endpoint(response) as endpoint:
        turn = await client.complete(messages=messages, tools=(definition,))

    request = endpoint.call_args.args[0]
    assert str(request.url) == "http://127.0.0.1:8000/v1/chat/completions"
    assert request.method == "POST"
    assert request.headers["Authorization"] == "Bearer test-key"
    payload = json.loads(request.content)
    assert set(payload) == {"model", "messages", "tools"}
    assert payload["model"] == "local-model"
    assert payload["messages"][1]["content"] == ""
    assert payload["messages"][1]["tool_calls"][0]["function"]["arguments"] == (
        '{"value":1}'
    )
    assert payload["messages"][2] == {
        "role": "tool",
        "content": '{"ok":true,"result":1}',
        "tool_call_id": "old-call",
        "name": "lookup",
    }
    assert payload["tools"][0]["function"] == {
        "name": "lookup",
        "description": "Look up one value.",
        "parameters": definition.parameters,
        "strict": True,
    }
    assert turn.finish_reason == "tool_calls"
    assert [(call.id, call.name, call.arguments) for call in turn.tool_calls] == [
        ("call-1", "lookup", {"value": 2}),
        ("call-2", "lookup", {"value": 3}),
    ]
    assert [call.purpose for call in turn.tool_calls] == [
        "Check the first value.",
        "Check the second value.",
    ]


async def test_replays_provider_call_and_nested_argument_order() -> None:
    arguments = (
        {
            "purpose": "Create the requested occurrence workbook.",
            "queries": [{
                "scope": {"dataset": "Birds"},
                "query": {
                    "filters": {"catalogNumber": {"exact_any": ["0007 — 雪"], "is_empty": False}},
                    "column_mode": "selected",
                    "return_columns": ["catalogNumber"],
                },
                "result": {"distinct_by": [], "order_by": [{"header": "catalogNumber", "direction": "desc"}]},
            }],
            "limit": 3,
            "limit_scope": "total",
            "layout": "single_workbook",
            "filename": "source.xlsx",
        },
        {
            "collections": ["Herbarium"],
            "purpose": "Check field availability for both scopes.",
            "datasets": ["birds"],
        },
    )
    raw_calls = [
        {
            "id": call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(value, ensure_ascii=False, separators=(",", ":")),
            },
        }
        for call_id, name, value in zip(
            ("call-z", "call-a"), ("generate_excel", "retrieve_schema"), arguments,
        )
    ]
    definitions = tuple(
        ToolDefinition(call["function"]["name"], "Biodiversity operation.", {"type": "object"})
        for call in raw_calls
    )
    client = OpenAIChatClient(base_url="http://localhost/v1", model="test")
    user_message = ChatMessage(role="user", content="Export records and retrieve more schema.")
    response = json_response({"choices": [{
        "finish_reason": "tool_calls",
        "message": {"role": "assistant", "content": None, "tool_calls": raw_calls},
    }]})
    finished = json_response({"choices": [{
        "message": {"role": "assistant", "content": "Finished."},
    }]})

    with mock_endpoint(response, finished) as endpoint:
        turn = await client.complete(messages=(user_message,), tools=definitions)
        await client.complete(
            messages=(
                user_message,
                ChatMessage(role="assistant", content=None, tool_calls=turn.tool_calls),
                *(ChatMessage(
                    role="tool", content='{"ok":true,"result":{}}',
                    tool_call_id=call.id, name=call.name,
                ) for call in turn.tool_calls),
            ),
            tools=definitions,
        )

    replayed = json.loads(endpoint.call_args.args[0].content)["messages"][1]["tool_calls"]
    assert replayed == raw_calls
    assert [call.id for call in turn.tool_calls] == ["call-z", "call-a"]


async def test_rejects_tool_call_without_public_purpose() -> None:
    document = {
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"value":2}'},
                }],
            },
        }],
    }

    with pytest.raises(OpenAIClientError, match="purpose must be a non-empty string"):
        await _complete_response(document)


async def test_generation_has_no_read_deadline_and_preserves_transport_defaults() -> None:
    async with httpx.AsyncClient() as default_client:
        default_timeouts = default_client.timeout.as_dict()
    client = OpenAIChatClient(base_url="http://localhost/v1", model="test")
    response = json_response(
        {"choices": [{"message": {"role": "assistant", "content": "Finished."}}]}
    )

    with mock_endpoint(response) as endpoint:
        await client.complete(
            messages=(ChatMessage(role="user", content="Take your time."),),
            tools=(),
        )

    request = endpoint.call_args.args[0]
    assert request.extensions["timeout"] == {**default_timeouts, "read": None}


@pytest.mark.parametrize("enable_thinking", (None, False, True))
async def test_context_count_uses_the_same_messages_tools_and_template_as_generation(
    enable_thinking: bool | None,
) -> None:
    client = OpenAIChatClient(
        base_url="https://example.test/inference/team/v1/", model="test", api_key="key"
    )
    call = ToolCall("call-1", "lookup", {"z": {"last": "雪", "first": 1}, "a": 2})
    messages = (
        ChatMessage(role="system", content="Find records."),
        ChatMessage(role="user", content="Seaweeds — 海藻"),
        ChatMessage(
            role="assistant", content=None, tool_calls=(call,),
            reasoning_content="Inspect the records.", reasoning_field="reasoning",
        ),
        ChatMessage(role="tool", content='{"count":2}', tool_call_id=call.id, name=call.name),
    )
    tools = (ToolDefinition("lookup", "Find matching records.", {"type": "object"}),)
    token_response = json_response({"count": 3, "max_model_len": 128_000, "tokens": [5, 6, 7]})
    generation_response = json_response({
        "choices": [{"message": {"role": "assistant", "content": "Two records."}}],
    })
    with mock_endpoint(token_response, generation_response) as endpoint:
        info = await client.context_info(
            messages=messages, tools=tools, enable_thinking=enable_thinking,
        )
        await client.complete(messages=messages, tools=tools, enable_thinking=enable_thinking)

    token_request, generation_request = [call.args[0] for call in endpoint.call_args_list]
    assert info == ContextInfo(prompt_tokens=3, context_limit_tokens=128_000)
    assert str(token_request.url) == "https://example.test/inference/team/tokenize"
    assert str(generation_request.url) == "https://example.test/inference/team/v1/chat/completions"
    assert token_request.headers["Authorization"] == "Bearer key"
    counted_payload = json.loads(token_request.content)
    generated_payload = json.loads(generation_request.content)
    assert counted_payload == {
        **generated_payload, "add_generation_prompt": True, "add_special_tokens": False,
    }
    assert counted_payload["messages"][2]["tool_calls"][0]["function"]["arguments"] == (
        '{"z":{"last":"雪","first":1},"a":2}'
    )
    if enable_thinking is None:
        assert "chat_template_kwargs" not in counted_payload
    else:
        assert counted_payload["chat_template_kwargs"] == {"enable_thinking": enable_thinking}
    assert "truncate_prompt_tokens" not in counted_payload


@pytest.mark.parametrize(
    ("base_url", "expected_url"),
    (
        ("http://localhost/v1", "http://localhost/tokenize"),
        ("http://localhost/proxy/v1/", "http://localhost/proxy/tokenize"),
        ("http://localhost/proxy", "http://localhost/proxy/tokenize"),
        ("http://localhost", "http://localhost/tokenize"),
    ),
)
async def test_standalone_text_count_uses_tokenizer_without_special_tokens(
    base_url: str, expected_url: str,
) -> None:
    client = OpenAIChatClient(base_url=base_url, model="test")
    with mock_endpoint(json_response({"count": 2, "max_model_len": 100, "tokens": [2, 3]})) as endpoint:
        count = await client.count_text_tokens("Algues — 海藻")

    request = endpoint.call_args.args[0]
    assert count == 2
    assert str(request.url) == expected_url
    assert json.loads(request.content) == {
        "model": "test", "prompt": "Algues — 海藻", "add_special_tokens": False,
    }


async def test_empty_text_counts_as_zero_without_special_tokens() -> None:
    client = OpenAIChatClient(base_url="http://localhost/v1", model="test")
    with mock_endpoint(json_response({"count": 0, "max_model_len": 100, "tokens": []})):
        assert await client.count_text_tokens("") == 0


async def test_counts_oversized_context_without_rejecting_or_truncating_it() -> None:
    client = OpenAIChatClient(base_url="http://localhost/v1", model="test")
    with mock_endpoint(json_response({"count": 3, "max_model_len": 2, "tokens": [1, 2, 3]})):
        info = await client.context_info(messages=(ChatMessage(role="user", content="Long input."),), tools=())
    assert info == ContextInfo(prompt_tokens=3, context_limit_tokens=2)


@pytest.mark.parametrize(
    ("document", "message"),
    (
        ([], "must be a JSON object"),
        ({"count": True, "max_model_len": 100, "tokens": [1]}, "count must be"),
        ({"count": -1, "max_model_len": 100, "tokens": []}, "count must be"),
        ({"max_model_len": 100, "tokens": []}, "count must be"),
        ({"count": 0, "max_model_len": True, "tokens": []}, "context limit must be"),
        ({"count": 0, "max_model_len": 0, "tokens": []}, "context limit must be"),
        ({"count": 0, "tokens": []}, "context limit must be"),
        ({"count": 1, "max_model_len": 100, "tokens": []}, "tokens must match"),
        ({"count": 1, "max_model_len": 100, "tokens": [True]}, "tokens must match"),
        ({"count": 1, "max_model_len": 100, "tokens": [-1]}, "tokens must match"),
        ({"count": 1, "max_model_len": 100}, "tokens must match"),
    ),
)
async def test_rejects_invalid_tokenizer_contract(document: object, message: str) -> None:
    client = OpenAIChatClient(base_url="http://localhost/v1", model="test")
    with mock_endpoint(json_response(document)):
        with pytest.raises(OpenAIClientError, match=message):
            await client.context_info(messages=(ChatMessage(role="user", content="Test."),), tools=())


@pytest.mark.parametrize(
    "response",
    (httpx.Response(503, text="private endpoint and file paths"), httpx.Response(404)),
)
async def test_tokenizer_failure_has_no_approximate_fallback_or_private_details(
    response: httpx.Response,
) -> None:
    client = OpenAIChatClient(base_url="http://private-backend/v1", model="test")
    with mock_endpoint(response) as endpoint:
        with pytest.raises(OpenAIClientError) as caught:
            await client.count_text_tokens("Secret user input.")
    assert str(caught.value) == "tokenization request failed: HTTPStatusError"
    assert endpoint.call_count == 1


@pytest.mark.parametrize("max_completion_tokens", (1, 16_000))
async def test_completion_sends_explicit_output_bound(max_completion_tokens: int) -> None:
    client = OpenAIChatClient(base_url="http://localhost/v1", model="test")
    response = json_response({
        "choices": [{"message": {"role": "assistant", "content": "Done."}}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    })
    with mock_endpoint(response) as endpoint:
        await client.complete(
            messages=(ChatMessage(role="user", content="Summarize."),), tools=(),
            max_completion_tokens=max_completion_tokens, enable_thinking=False,
        )
    payload = json.loads(endpoint.call_args.args[0].content)
    assert payload["max_completion_tokens"] == max_completion_tokens
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "max_tokens" not in payload
    assert "truncate_prompt_tokens" not in payload


@pytest.mark.parametrize("bound", (True, 0, -1, 1.5, "100"))
async def test_invalid_completion_bound_is_rejected_before_request(bound: object) -> None:
    client = OpenAIChatClient(base_url="http://localhost/v1", model="test")
    with mock_endpoint() as endpoint:
        with pytest.raises(ValueError, match="positive integer"):
            await client.complete(messages=(), tools=(), max_completion_tokens=bound)
    assert endpoint.call_count == 0


async def test_rejects_reported_generation_above_output_bound() -> None:
    client = OpenAIChatClient(base_url="http://localhost/v1", model="test")
    response = json_response({
        "choices": [{"message": {"role": "assistant", "content": "Too long."}}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 11, "total_tokens": 13},
    })
    with mock_endpoint(response):
        with pytest.raises(OpenAIClientError, match="exceeded its output token limit"):
            await client.complete(messages=(), tools=(), max_completion_tokens=10)


async def test_parses_usage_response_id_and_vllm_timing_metrics() -> None:
    turn = await _complete_response(
        {
            "id": "chatcmpl-request-123",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Finished."},
                }
            ],
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "total_tokens": 150,
                "prompt_tokens_details": {
                    "cached_tokens": 80,
                    "provider_extension": "ignored",
                },
                "completion_tokens_details": {
                    "reasoning_tokens": 12,
                    "accepted_prediction_tokens": "ignored",
                },
            },
            "metrics": {
                "time_to_first_token_ms": 250.5,
                "generation_time_ms": 1_500,
                "queue_time_ms": 10,
                "mean_itl_ms": 50,
                "tokens_per_second": 20,
                "provider_extension": "ignored",
            },
        }
    )

    assert turn.response_id == "chatcmpl-request-123"
    assert turn.usage == TokenUsage(
        prompt_tokens=120,
        completion_tokens=30,
        total_tokens=150,
        cached_prompt_tokens=80,
        reasoning_tokens=12,
    )
    assert turn.provider_metrics == ModelRequestMetrics(
        provider="vllm",
        time_to_first_token_seconds=0.2505,
        generation_seconds=1.5,
        queue_seconds=0.01,
        mean_inter_token_latency_seconds=0.05,
        tokens_per_second=20,
    )


@pytest.mark.parametrize(
    "usage",
    (
        [],
        {"prompt_tokens": 1, "total_tokens": 1},
        {"prompt_tokens": True, "completion_tokens": 1, "total_tokens": 2},
        {"prompt_tokens": -1, "completion_tokens": 1, "total_tokens": 0},
        {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 3},
    ),
)
async def test_malformed_usage_does_not_discard_a_valid_answer(usage: object) -> None:
    turn = await _complete_response({
        "choices": [{"message": {"role": "assistant", "content": "Finished."}}],
        "usage": usage,
    })
    assert turn.content == "Finished."
    assert turn.usage is None


async def test_malformed_token_details_preserve_valid_usage_and_details() -> None:
    turn = await _complete_response({
        "choices": [{"message": {"role": "assistant", "content": "Finished."}}],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
            "prompt_tokens_details": {"cached_tokens": 2},
            "completion_tokens_details": {"reasoning_tokens": 1},
        },
    })
    assert turn.usage == TokenUsage(1, 2, 3, reasoning_tokens=1)


@pytest.mark.parametrize("invalid_metric", (-1, True, "10", float("nan"), float("inf"), 10**400))
async def test_malformed_metrics_preserve_other_usable_metrics(invalid_metric: object) -> None:
    turn = await _complete_response({
        "choices": [{"message": {"role": "assistant", "content": "Finished."}}],
        "metrics": {"queue_time_ms": invalid_metric, "tokens_per_second": 20},
    })
    assert turn.content == "Finished."
    assert turn.provider_metrics == ModelRequestMetrics(provider="vllm", tokens_per_second=20)


async def test_provider_metadata_has_no_arbitrary_numeric_or_id_caps() -> None:
    tokens = 2**64
    turn = await _complete_response({
        "id": "x" * 300,
        "choices": [{"message": {"role": "assistant", "content": "Finished."}}],
        "usage": {"prompt_tokens": tokens, "completion_tokens": 1, "total_tokens": tokens + 1},
        "metrics": {"tokens_per_second": 1e10},
    })
    assert turn.response_id == "x" * 300
    assert turn.usage == TokenUsage(tokens, 1, tokens + 1)
    assert turn.provider_metrics == ModelRequestMetrics(provider="vllm", tokens_per_second=1e10)


@pytest.mark.parametrize(
    ("content", "finish_reason", "error"),
    (
        ([], "stop", "content must be a string or None"),
        ("Finished.", {}, "finish reason must be a string or None"),
        (None, "stop", "must contain content, reasoning, or at least one tool call"),
    ),
)
async def test_rejects_invalid_assistant_fields_at_the_model_turn_boundary(
    content: object,
    finish_reason: object,
    error: str,
) -> None:
    document = {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": content},
            }
        ]
    }

    with pytest.raises(OpenAIClientError, match=error):
        await _complete_response(document)


@pytest.mark.parametrize(
    "metrics",
    (None, [], {}, {"queue_time_ms": None, "provider_extension": 123}),
)
async def test_omits_unavailable_metrics(metrics: object) -> None:
    turn = await _complete_response(
        {
            "choices": [{"message": {"role": "assistant", "content": "Done."}}],
            "metrics": metrics,
        }
    )

    assert turn.provider_metrics is None


async def test_preserves_zero_metrics_and_token_details() -> None:
    turn = await _complete_response(
        {
            "choices": [{"message": {"role": "assistant", "content": "Done."}}],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens_details": None,
            },
            "metrics": {"queue_time_ms": 0, "tokens_per_second": 0},
        }
    )

    assert turn.usage == TokenUsage(1, 1, 2, cached_prompt_tokens=0)
    assert turn.provider_metrics == ModelRequestMetrics(
        provider="vllm", queue_seconds=0.0, tokens_per_second=0.0
    )


async def test_rejects_malformed_tool_call_arguments() -> None:
    document = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": "[]",
                            },
                        }
                    ],
                }
            }
        ]
    }

    with pytest.raises(OpenAIClientError, match="JSON object"):
        await _complete_response(document)


async def test_rejects_provider_response_without_assistant_role() -> None:
    document = {
        "choices": [
            {
                "message": {"content": "Finished."},
            }
        ]
    }

    with pytest.raises(OpenAIClientError, match="role must be assistant"):
        await _complete_response(document)


async def test_accepts_valid_response_larger_than_two_mebibytes() -> None:
    content = "海藻" * 350_000
    response = json_response(
        {"choices": [{"message": {"role": "assistant", "content": content}}]}
    )
    assert len(response.content) > 2_097_152
    client = OpenAIChatClient(base_url="http://localhost/v1", model="test")

    with mock_endpoint(response):
        turn = await client.complete(
            messages=(ChatMessage(role="user", content="Return the full response."),),
            tools=(),
        )

    assert turn.content == content


async def test_accepts_utf8_response_without_provider_metadata() -> None:
    content = "Algues — 海藻"
    response = json_response(
        {"choices": [{"message": {"role": "assistant", "content": content}}]}
    )
    client = OpenAIChatClient(
        base_url="http://localhost/v1",
        model="test",
    )

    with mock_endpoint(response) as endpoint:
        turn = await client.complete(
            messages=(ChatMessage(role="user", content=content),),
            tools=(),
        )

    payload = json.loads(endpoint.call_args.args[0].content)
    assert set(payload) == {"model", "messages"}
    assert payload["messages"][0]["content"] == content
    assert turn == ModelTurn(content=content)


@pytest.mark.parametrize(
    ("body", "error_message"),
    ((b"\xff", "not UTF-8"), (b"{", "not valid JSON")),
)
async def test_rejects_invalid_response_json(body: bytes, error_message: str) -> None:
    client = OpenAIChatClient(base_url="http://localhost/v1", model="test")
    with mock_endpoint(httpx.Response(200, content=body)):
        with pytest.raises(OpenAIClientError, match=error_message):
            await client.complete(
                messages=(ChatMessage(role="user", content="Hello."),),
                tools=(),
            )


@pytest.mark.parametrize("response_field", ("reasoning_content", "reasoning"))
async def test_preserves_explicit_reasoning_and_round_trips_supported_alias(
    response_field: str,
) -> None:
    first_response = json_response(
        {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        response_field: "I should call the lookup tool.",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "lookup",
                                    "arguments": '{"purpose":"Check the requested value.","value":2}',
                                },
                            }
                        ],
                    },
                }
            ]
        }
    )
    second_response = json_response(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Finished."},
                }
            ]
        }
    )
    client = OpenAIChatClient(base_url="http://localhost/v1", model="test")

    with mock_endpoint(first_response, second_response) as endpoint:
        first_turn = await client.complete(
            messages=(ChatMessage(role="user", content="Use a tool."),),
            tools=(),
        )
        await client.complete(
            messages=(
                ChatMessage(role="user", content="Use a tool."),
                ChatMessage(
                    role="assistant",
                    content=first_turn.content,
                    reasoning_content=first_turn.reasoning_content,
                    reasoning_field=first_turn.reasoning_field,
                    tool_calls=first_turn.tool_calls,
                ),
                ChatMessage(
                    role="tool",
                    content='{"ok":true,"result":2}',
                    tool_call_id="call-1",
                    name="lookup",
                ),
            ),
            tools=(),
        )

    assert first_turn.reasoning_content == "I should call the lookup tool."
    second_request = json.loads(endpoint.call_args_list[1].args[0].content)
    assert second_request["messages"][1][response_field] == (
        "I should call the lookup tool."
    )
    other_alias = (
        "reasoning" if response_field == "reasoning_content" else "reasoning_content"
    )
    assert other_alias not in second_request["messages"][1]


async def test_rejects_conflicting_reasoning_aliases() -> None:
    document = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Finished.",
                    "reasoning_content": "first",
                    "reasoning": "second",
                }
            }
        ]
    }

    with pytest.raises(OpenAIClientError, match="conflicting"):
        await _complete_response(document)


@pytest.mark.parametrize(
    "content",
    (
        'I will call it. <tool_call>{"name":"lookup"}',
        "Unexpected closing tag: </tool_call>",
    ),
)
async def test_rejects_unparsed_tool_call_envelopes(content: str) -> None:
    document = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ]
    }

    with pytest.raises(OpenAIClientError, match="unparsed tool-call"):
        await _complete_response(document)


async def _respond_after_release(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    slow_started: asyncio.Event,
    release_slow: asyncio.Event,
) -> None:
    try:
        headers = (await reader.readuntil(b"\r\n\r\n")).decode("ascii")
        length = next(
            int(line.split(":", 1)[1])
            for line in headers.splitlines()
            if line.lower().startswith("content-length:")
        )
        request = json.loads(await reader.readexactly(length))
        query = request["messages"][0]["content"]
        if query == "slow":
            slow_started.set()
            await release_slow.wait()
        body = json_response({
            "choices": [{"message": {"role": "assistant", "content": query}}],
        }).content
        writer.write(
            f"HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
            + body
        )
        await writer.drain()
    finally:
        writer.close()
        await writer.wait_closed()


async def test_faster_http_request_finishes_while_another_request_is_pending() -> None:
    slow_started = asyncio.Event()
    release_slow = asyncio.Event()

    respond = partial(
        _respond_after_release, slow_started=slow_started, release_slow=release_slow,
    )
    server = await asyncio.start_server(respond, "127.0.0.1", 0)
    async with server:
        port = server.sockets[0].getsockname()[1]
        client = OpenAIChatClient(base_url=f"http://127.0.0.1:{port}/v1", model="test")
        slow = asyncio.create_task(client.complete(
            messages=(ChatMessage(role="user", content="slow"),), tools=(),
        ))
        try:
            await asyncio.wait_for(slow_started.wait(), timeout=5)
            fast = await asyncio.wait_for(client.complete(
                messages=(ChatMessage(role="user", content="fast"),), tools=(),
            ), timeout=5)
            assert fast.content == "fast"
            assert not slow.done()
        finally:
            release_slow.set()
            slow_result = await asyncio.wait_for(slow, timeout=5)
        assert slow_result.content == "slow"


async def test_reasoning_aliases_stay_with_their_conversation() -> None:
    aliases = ("reasoning", "reasoning_content")
    responses = [json_response({
        "choices": [{"message": {"role": "assistant", "content": "Answer", alias: alias}}],
    }) for alias in aliases]
    client = OpenAIChatClient(base_url="http://localhost/v1", model="test")
    with mock_endpoint(*responses, json_response({"choices": [{
        "message": {"role": "assistant", "content": "Continued."},
    }]})) as endpoint:
        turns = await asyncio.gather(*(client.complete(
            messages=(ChatMessage(role="user", content=alias),), tools=(),
        ) for alias in aliases))
        first_turn = turns[0]
        await client.complete(messages=(ChatMessage(
            role="assistant", content=first_turn.content,
            reasoning_content=first_turn.reasoning_content,
            reasoning_field=first_turn.reasoning_field,
        ), ChatMessage(role="user", content="Continue.")), tools=())

    continued = json.loads(endpoint.call_args.args[0].content)["messages"][0]
    assert continued["reasoning"] == "reasoning"
    assert "reasoning_content" not in continued


async def test_transport_errors_do_not_expose_the_response_body() -> None:
    client = OpenAIChatClient(base_url="http://localhost/v1", model="test")
    with mock_endpoint(httpx.Response(503, content=b"PRIVATE provider detail")):
        with pytest.raises(OpenAIClientError, match="HTTPStatusError") as captured:
            await client.complete(messages=(ChatMessage(role="user", content="Try."),), tools=())
    assert "PRIVATE" not in str(captured.value)
