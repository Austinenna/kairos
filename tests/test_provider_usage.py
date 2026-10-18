"""验证用量来源、未知值和费用估算在两种请求模式间保持一致。"""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from openai.types import CompletionUsage

from kairos.pricing import PRICES, Usage, estimate_cost, usage_from_openai
from kairos.providers.openai_compatible import OpenAICompatibleModel, normalize_openai_usage


class FakeStream:
    def __init__(self, usage):
        self.usage = usage
        self.closed = False

    def __iter__(self):
        yield {"choices": [{"index": 0, "delta": {"content": "完成"}, "finish_reason": "stop"}]}
        yield {"choices": [], "usage": self.usage}

    def close(self):
        self.closed = True


@pytest.mark.parametrize("sdk_object", [False, True])
def test_complete_and_stream_share_all_usage_details(sdk_object):
    raw_usage = {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "prompt_tokens_details": {"cached_tokens": 80},
        "completion_tokens_details": {"reasoning_tokens": 12},
    }
    usage = CompletionUsage.model_validate(raw_usage) if sdk_object else raw_usage
    response_stream = FakeStream(usage)

    def create(**request):
        if request.get("stream"):
            return response_stream
        return {
            "choices": [{"message": {"content": "完成"}, "finish_reason": "stop"}],
            "usage": usage,
        }

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    model = OpenAICompatibleModel(model="test-model", client=client)
    complete = model.complete(messages=[], tools=[])
    streamed = model.stream(messages=[], tools=[], on_text=lambda _: None)

    assert (
        complete
        == streamed
        == {
            "content": "完成",
            "tool_calls": [],
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "cached_input_tokens": 80,
                "reasoning_tokens": 12,
            },
        }
    )
    assert response_stream.closed
    assert usage_from_openai(usage) == Usage(100, 80, 20, 12)


@pytest.mark.parametrize(
    "fields,expected",
    [
        ({}, None),
        ({"prompt_cache_hit_tokens": 7}, 7),
        ({"prompt_tokens_details": {"prompt_cache_hit_tokens": 6}}, 6),
        ({"prompt_tokens_details": {"cached_tokens": 5}}, 5),
        (
            {
                "prompt_cache_hit_tokens": 0,
                "prompt_tokens_details": {"prompt_cache_hit_tokens": 6, "cached_tokens": 5},
            },
            0,
        ),
        ({"prompt_tokens_details": {"prompt_cache_hit_tokens": 0, "cached_tokens": 5}}, 0),
        (
            {
                "prompt_cache_hit_tokens": None,
                "prompt_tokens_details": {"prompt_cache_hit_tokens": None, "cached_tokens": 0},
            },
            0,
        ),
        ({"prompt_cache_hit_tokens": -1, "prompt_tokens_details": {"cached_tokens": 5}}, None),
        ({"prompt_tokens_details": {"prompt_cache_hit_tokens": False, "cached_tokens": 5}}, None),
    ],
)
def test_cache_source_precedence_distinguishes_zero_missing_and_invalid(fields, expected):
    raw = {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13, **fields}
    converted = normalize_openai_usage(raw)
    legacy = usage_from_openai(raw)
    assert converted["cached_input_tokens"] == legacy.cached_input_tokens == expected
    assert converted["reasoning_tokens"] is legacy.reasoning_tokens is None


def test_missing_usage_fields_remain_unknown_and_explicit_zero_is_preserved():
    assert normalize_openai_usage({}) == {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "cached_input_tokens": None,
        "reasoning_tokens": None,
    }
    assert usage_from_openai(None) == Usage(None, None, None, None)
    assert normalize_openai_usage(
        {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "prompt_cache_hit_tokens": 0,
            "completion_tokens_details": {"reasoning_tokens": 0},
        }
    ) == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cached_input_tokens": 0,
        "reasoning_tokens": 0,
    }


@pytest.mark.parametrize("invalid", [True, False, -1, "12", 1.5, float("nan"), [], {}])
def test_invalid_counts_are_unknown_without_conversion_errors(invalid):
    result = normalize_openai_usage(
        {
            "prompt_tokens": invalid,
            "completion_tokens": invalid,
            "total_tokens": invalid,
            "prompt_cache_hit_tokens": invalid,
            "completion_tokens_details": {"reasoning_tokens": invalid},
        }
    )
    assert all(value is None for value in result.values())


@pytest.mark.parametrize("stream", [False, True])
def test_no_provider_usage_does_not_become_zero_usage(stream):
    response_stream = FakeStream(None)
    response = {"choices": [{"message": {"content": "完成"}, "finish_reason": "stop"}]}
    endpoint = SimpleNamespace(create=lambda **_: response_stream if stream else response)
    model = OpenAICompatibleModel(
        model="test-model", client=SimpleNamespace(chat=SimpleNamespace(completions=endpoint))
    )
    result = (
        model.stream(messages=[], tools=[], on_text=lambda _: None) if stream else model.complete(messages=[], tools=[])
    )
    assert "usage" not in result


@pytest.mark.parametrize("unknown", ["input_tokens", "cached_input_tokens", "output_tokens"])
def test_cost_is_unknown_when_a_required_count_is_missing(unknown):
    fields = {"input_tokens": 100, "cached_input_tokens": 80, "output_tokens": 20, "reasoning_tokens": 12}
    fields[unknown] = None
    assert estimate_cost("deepseek-flash", Usage(**fields)) is None


def test_cost_uses_cache_split_and_does_not_require_reasoning_details():
    off_peak = datetime(2026, 10, 18, 0, 0, tzinfo=UTC)
    hit, miss, output = PRICES["deepseek-flash"]["off_peak"]
    unknown_reasoning = Usage(1_000_000, 800_000, 1_000_000, None)
    known_reasoning = Usage(1_000_000, 800_000, 1_000_000, 200_000)

    assert estimate_cost("deepseek-flash", unknown_reasoning, off_peak) == pytest.approx(
        0.8 * hit + 0.2 * miss + output
    )
    assert estimate_cost("deepseek-flash", unknown_reasoning, off_peak) == estimate_cost(
        "deepseek-flash", known_reasoning, off_peak
    )
    assert estimate_cost("deepseek-flash", Usage(0, 0, 0, None), off_peak) == 0
    assert estimate_cost("unknown-model", known_reasoning, off_peak) is None
