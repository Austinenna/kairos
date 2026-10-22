from types import SimpleNamespace as NS

from kairos.models.anthropic import to_anthropic_messages
from kairos.models.openai_compat import OpenAICompatibleModel


class FakeStream(list):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def chunk(content=None, reasoning=None, tool_calls=None, finish_reason=None, usage=None):
    delta = NS(content=content, reasoning_content=reasoning, tool_calls=tool_calls)
    choices = [NS(delta=delta, finish_reason=finish_reason)] if usage is None else []
    return NS(choices=choices, usage=usage)


def tool_delta(index, id=None, name=None, arguments=None):
    return NS(index=index, id=id, function=NS(name=name, arguments=arguments))


def test_openai_stream_assembles_text_tool_calls_and_usage():
    chunks = FakeStream(
        [
            chunk(reasoning="想一想"),
            chunk(content="好的"),
            chunk(tool_calls=[tool_delta(0, id="c1", name="add_todo", arguments='{"title":')]),
            chunk(tool_calls=[tool_delta(0, arguments=' "买牛奶"}')], finish_reason="tool_calls"),
            chunk(usage=NS(prompt_tokens=10, completion_tokens=5, prompt_cache_hit_tokens=8)),
        ]
    )
    client = NS(chat=NS(completions=NS(create=lambda **request: chunks)))
    model = OpenAICompatibleModel("deepseek-flash", client=client)
    shown = []

    response = model.stream([{"role": "user", "content": "hi"}], [], shown.append)

    assert shown == ["好的"]
    assert response.tool_calls == [
        {"id": "c1", "type": "function", "function": {"name": "add_todo", "arguments": '{"title": "买牛奶"}'}}
    ]
    assert response.extra == {"reasoning_content": "想一想"}
    assert (response.usage.input_tokens, response.usage.cached_input_tokens) == (10, 8)


def test_anthropic_conversion_groups_tool_results_and_replays_raw_blocks():
    raw_blocks = [{"type": "thinking", "thinking": "…", "signature": "sig"}, {"type": "tool_use", "id": "t1"}]
    messages = [
        {"role": "system", "content": "规则"},
        {"role": "user", "content": "问题"},
        {"role": "assistant", "content": "", "provider_content": raw_blocks, "tool_calls": []},
        {"role": "tool", "tool_call_id": "t1", "content": "1"},
        {"role": "tool", "tool_call_id": "t2", "content": "2"},
    ]

    system, converted = to_anthropic_messages(messages)

    assert system == "规则"
    assert converted[1] == {"role": "assistant", "content": raw_blocks}
    assert [block["tool_use_id"] for block in converted[2]["content"]] == ["t1", "t2"]
