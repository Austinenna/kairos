"""通过实际工具续轮验证累计用量、最新请求输入和展示口径。"""

from copy import deepcopy
from io import StringIO

import pytest
from rich.console import Console

from kairos import pricing
from kairos.context import REFERENCE_HEADER
from kairos.runtime import create_runtime
from kairos.terminal_renderer import TerminalRenderer


class MeteredModel:
    model = "usage-test"

    def __init__(self, first_usage, last_usage):
        self.responses = iter(
            [
                {
                    "tool_calls": [{"id": "list", "function": {"name": "list_todos", "arguments": "{}"}}],
                    "usage": first_usage,
                },
                {"content": "没有待办", "usage": last_usage},
            ]
        )
        self.requests = []

    def complete(self, *, messages, tools):
        self.requests.append(deepcopy(messages))
        return next(self.responses)


FIRST = {"prompt_tokens": 6000, "completion_tokens": 100, "cached_input_tokens": 0, "reasoning_tokens": 10}
LAST = {"prompt_tokens": 7000, "completion_tokens": 300, "cached_input_tokens": 0, "reasoning_tokens": 20}


def render(result, monkeypatch):
    monkeypatch.setitem(pricing.CONTEXT_WINDOW, "usage-test", 10000)
    monkeypatch.setitem(pricing.PRICES, "usage-test", {"peak": (1, 2, 3), "off_peak": (1, 2, 3)})
    output = StringIO()
    renderer = TerminalRenderer(Console(file=output, width=220, color_system=None))
    cost = renderer.render_turn(result, model="usage-test", total_cost=0)
    return output.getvalue(), cost


def test_tool_turn_counts_all_requests_but_context_uses_latest_full_input(tmp_path, monkeypatch):
    model = MeteredModel(FIRST, LAST)
    with create_runtime(model=model, db_path=tmp_path / "usage.sqlite3") as runtime:
        result = runtime.respond("查询待办")
    assert model.requests[1][: len(model.requests[0])] == model.requests[0]
    assert [message["role"] for message in model.requests[1]] == ["system", "user", "user", "assistant", "tool"]
    for request in model.requests:
        assert sum(str(message.get("content", "")).startswith(REFERENCE_HEADER) for message in request) == 1
    assert [message["role"] for message in result.messages] == ["system", "user", "assistant", "tool", "assistant"]
    assert result.usage.input_tokens == 13000
    assert result.usage.output_tokens == 400
    assert result.last_request_usage.input_tokens == 7000
    assert result.last_request_usage.output_tokens == 300
    text, cost = render(result, monkeypatch)
    assert "累计输入 13,000" in text
    assert "最近请求输入 7,000 / 10,000（70.00%，不含新输出）" in text
    assert "130.00%" not in text
    assert cost == pytest.approx((13000 * 2 + 400 * 3) / 1_000_000)


@pytest.mark.parametrize("missing", ["first", "last", "both"])
def test_missing_usage_does_not_become_zero_or_stale_context(tmp_path, monkeypatch, missing):
    model = MeteredModel(None if missing != "last" else FIRST, None if missing != "first" else LAST)
    with create_runtime(model=model, db_path=tmp_path / "usage.sqlite3") as runtime:
        result = runtime.respond("查询待办")
    assert result.usage is None or result.usage.input_tokens is None
    if missing in {"last", "both"}:
        assert result.last_request_usage is None
    else:
        assert result.last_request_usage.input_tokens == 7000
    text, _ = render(result, monkeypatch)
    assert "本轮费用未估算" in text
    assert "已估算累计" in text
    assert "累计输入 0" not in text
    if missing != "first":
        assert "最近请求输入未知" in text
        assert "%" not in text


@pytest.mark.parametrize("unknown", [None, True, -1, "7000"])
def test_invalid_or_missing_input_count_is_unknown_without_breaking_completed_turn(tmp_path, monkeypatch, unknown):
    model = MeteredModel(FIRST, {**LAST, "prompt_tokens": unknown})
    with create_runtime(model=model, db_path=tmp_path / "usage.sqlite3") as runtime:
        result = runtime.respond("查询待办")
        assert runtime.session_details()["turns"][0]["status"] == "completed"
    assert result.usage.input_tokens is None
    assert result.usage.output_tokens == 400
    assert result.last_request_usage.input_tokens is None
    text, _ = render(result, monkeypatch)
    assert "累计输入 未知" in text
    assert "最近请求输入未知" in text
    assert "本轮费用未估算" in text


def test_unknown_cache_keeps_known_input_but_does_not_estimate_cost(tmp_path, monkeypatch):
    model = MeteredModel(FIRST, {"prompt_tokens": 7000, "completion_tokens": 300})
    with create_runtime(model=model, db_path=tmp_path / "usage.sqlite3") as runtime:
        result = runtime.respond("查询待办")
    assert result.usage.cached_input_tokens is None
    assert result.usage.reasoning_tokens is None
    text, _ = render(result, monkeypatch)
    assert "缓存命中 未知" in text
    assert "70.00%" in text
    assert "本轮费用未估算" in text
