import json

from kairos.memory.extractor import MemoryExtractor, recent_dialogue
from kairos.memory.store import MemoryStore
from kairos.models import ModelResponse


def test_store_core_and_topics(tmp_path):
    store = MemoryStore(tmp_path)

    store.add_core("用户叫 Austin")
    assert store.write_topic("职业背景", "程序员", purpose="调整技术话题的深度") == "created"
    assert store.write_topic("职业背景", "在学 Agent 开发") == "appended"

    assert "- 用户叫 Austin" in store.core()
    assert "程序员\n\n在学 Agent 开发" in store.read_topic("职业背景")
    assert "- 职业背景：调整技术话题的深度" in store.snapshot()


def test_extractor_writes_topic_in_background(tmp_path):
    store = MemoryStore(tmp_path)
    arguments = {"title": "学习方向", "addition": "在学 Agent 开发", "purpose": "推荐学习资料时参考"}
    call = {"id": "c1", "type": "function", "function": {"name": "write_topic", "arguments": json.dumps(arguments)}}
    responses = [ModelResponse(content="", tool_calls=[call]), ModelResponse(content="无")]

    class FakeModel:
        def stream(self, messages, tools, on_text=None):
            return responses.pop(0)

    extractor = MemoryExtractor(store, lambda: FakeModel())
    extractor.submit([{"role": "user", "content": "我在学 Agent 开发"}, {"role": "assistant", "content": "好的"}])
    extractor.close()

    assert "在学 Agent 开发" in store.read_topic("学习方向")
    assert extractor.drain_notices() == ["已记住：主题“学习方向”"]


def test_recent_dialogue_marks_current_turn():
    messages = [{"role": "system", "content": "规则"}]
    for number in range(1, 5):
        messages += [{"role": "user", "content": f"问{number}"}, {"role": "assistant", "content": f"答{number}"}]

    text = recent_dialogue(messages, turns=2)

    assert text == "【之前】\n用户：问3\n助手：答3\n\n【本轮】\n用户：问4\n助手：答4"
