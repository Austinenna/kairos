"""用真实适配器和离线客户端验证思考开关的参数、保存及恢复边界。"""

import copy
import sqlite3
from types import SimpleNamespace

import pytest

import kairos.runtime as runtime_module
from kairos import cli
from kairos.config import Settings
from kairos.providers import AnthropicMessagesModel, OpenAICompatibleModel
from kairos.runtime import create_runtime


class OfflineClient:
    def __init__(self):
        self.requests = []
        self.closed = False
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.openai_create))
        self.messages = SimpleNamespace(create=self.anthropic_create)

    def openai_create(self, **request):
        self.requests.append(copy.deepcopy(request))
        return {"choices": [{"message": {"content": "已收到"}, "finish_reason": "stop"}]}

    def anthropic_create(self, **request):
        self.requests.append(copy.deepcopy(request))
        return {"content": [{"type": "text", "text": "已收到"}], "stop_reason": "end_turn"}

    def close(self):
        self.closed = True


class CustomModel:
    # 只使用相同前缀，不能因此假装已实现对应协议。
    model = "deepseek-custom"

    def complete(self, **request):
        raise AssertionError("校验开关不应请求模型")


class CaptureConsole:
    def __init__(self, inputs):
        self.inputs = iter(inputs)
        self.output = []

    def input(self, prompt):
        return next(self.inputs)

    def print(self, *values, **kwargs):
        self.output.append(" ".join(str(value) for value in values))


def unsupported_model(kind):
    if kind == "anthropic":
        return AnthropicMessagesModel("claude-test", client=OfflineClient())
    if kind == "openai":
        return OpenAICompatibleModel(model="other-model", client=OfflineClient())
    return CustomModel()


def managed_runtime(tmp_path, monkeypatch):
    built_models = []
    original_build = runtime_module.build_model

    def build(settings, client=None, *, thinking_enabled=False):
        assert client is None
        model = original_build(settings, client=OfflineClient(), thinking_enabled=thinking_enabled)
        built_models.append(model)
        return model

    monkeypatch.setattr(runtime_module, "build_model", build)
    settings = Settings(_env_file=None, gateway_base_url="https://offline.invalid/v1", kairos_model="deepseek-test")
    # 此组验证主模型的适配器切换；后台提取工厂由独立记忆集成测试覆盖。
    return create_runtime(settings=settings, db_path=tmp_path / "sessions.sqlite3", auto_memory=False), built_models


def test_thinking_toggle_changes_requests_and_persisted_state_without_losing_extra_options(tmp_path):
    client = OfflineClient()
    model = OpenAICompatibleModel(model="deepseek-test", client=client, extra_body={"other_option": "retained"})
    with create_runtime(model=model, db_path=tmp_path / "sessions.sqlite3", thinking_enabled=True) as runtime:
        assert runtime.supports_thinking
        runtime.respond("开启时请求")
        runtime.set_thinking(False)
        assert not runtime.thinking_enabled
        assert not runtime.session_info.thinking_enabled
        runtime.respond("关闭时请求")
        runtime.set_thinking(True)
        assert runtime.thinking_enabled
        assert runtime.session_info.thinking_enabled
        assert model.extra_body == {"other_option": "retained", "thinking": {"type": "enabled"}}
    assert [request["extra_body"] for request in client.requests] == [
        {"other_option": "retained", "thinking": {"type": "enabled"}},
        {"other_option": "retained", "thinking": {"type": "disabled"}},
    ]


@pytest.mark.parametrize("kind", ["anthropic", "openai", "custom"])
def test_unintegrated_models_reject_manual_toggle_without_saving_or_requesting(tmp_path, kind):
    model = unsupported_model(kind)
    with create_runtime(model=model, db_path=tmp_path / "sessions.sqlite3") as runtime:
        before = runtime.session_info
        assert not runtime.supports_thinking
        for enabled in (True, False):
            with pytest.raises(ValueError, match="尚未为当前模型接入思考开关"):
                runtime.set_thinking(enabled)
            assert not runtime.thinking_enabled
            assert runtime.session_info == before
        if hasattr(model, "client"):
            assert model.client.requests == []


@pytest.mark.parametrize("kind", ["anthropic", "openai", "custom"])
def test_unintegrated_model_cannot_create_enabled_session(tmp_path, kind):
    database = tmp_path / "sessions.sqlite3"
    with pytest.raises(ValueError, match="无法创建开启思考的会话"):
        create_runtime(model=unsupported_model(kind), db_path=database, thinking_enabled=True)
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0


@pytest.mark.parametrize("kind", ["anthropic", "openai", "custom"])
def test_startup_corrects_legacy_unintegrated_thinking_and_cli_explains_it(tmp_path, kind):
    database = tmp_path / "sessions.sqlite3"
    model = unsupported_model(kind)
    with create_runtime(model=model, db_path=database) as runtime:
        session_id = runtime.session_id
        runtime.rename_session("旧会话")
        runtime.session_store.update_thinking(session_id, True)

    with create_runtime(model=model, db_path=database, session_id=session_id) as runtime:
        assert not runtime.thinking_enabled
        assert not runtime.session_info.thinking_enabled
        assert runtime.thinking_state_corrected
        console = CaptureConsole(["/think", "/exit"])
        cli.run(console=console, runtime=runtime)
        output = "\n".join(console.output)
        assert "思考切换未接入" in output
        assert "旧会话的开启状态已纠正为关闭" in output
        assert "思考模式：开" not in output
        runtime.resume_session(session_id)
        assert not runtime.thinking_state_corrected

    with create_runtime(model=model, db_path=database, session_id=session_id) as runtime:
        assert not runtime.thinking_enabled
        assert not runtime.thinking_state_corrected


def test_cross_provider_resume_uses_target_capability_and_preserves_previous_model_options(tmp_path, monkeypatch):
    runtime, models = managed_runtime(tmp_path, monkeypatch)
    with runtime:
        runtime.set_thinking(True)
        original_id = runtime.session_id
        previous_model = runtime.model
        target = runtime.session_store.create_session(
            "旧 Anthropic 会话", provider="anthropic", model="claude-test", thinking_enabled=True, title="旧会话"
        )
        console = CaptureConsole([f"/resume {target.session_id}", "/exit"])
        cli.run(console=console, runtime=runtime)
        assert isinstance(runtime.model, AnthropicMessagesModel)
        assert not runtime.supports_thinking
        assert not runtime.thinking_enabled
        assert not runtime.session_info.thinking_enabled
        assert runtime.thinking_state_corrected
        assert "旧会话的开启状态已纠正为关闭" in "\n".join(console.output)
        assert previous_model.extra_body["thinking"]["type"] == "enabled"
        assert previous_model.client.closed
        runtime.respond("继续 Anthropic")
        anthropic_request = runtime.model.client.requests[-1]
        assert "thinking" not in anthropic_request
        assert "extra_body" not in anthropic_request

        runtime.resume_session(original_id)
        assert isinstance(runtime.model, OpenAICompatibleModel)
        assert runtime.supports_thinking
        assert runtime.thinking_enabled
        assert runtime.session_info.thinking_enabled
        assert not runtime.thinking_state_corrected
        runtime.respond("继续 DeepSeek")
        assert runtime.model.client.requests[-1]["extra_body"]["thinking"]["type"] == "enabled"
    assert len(models) == 3
    assert all(model.client.closed for model in models)


@pytest.mark.parametrize("failure_point", ["load_messages", "update_thinking", "touch_session"])
def test_failed_resume_keeps_original_active_model_session_and_thinking(tmp_path, monkeypatch, failure_point):
    runtime, models = managed_runtime(tmp_path, monkeypatch)
    with runtime:
        runtime.set_thinking(True)
        runtime.respond("原会话")
        original_model = runtime.model
        original_id = runtime.session_id
        original_messages = copy.deepcopy(runtime.messages)
        original_result = runtime.last_result
        target = runtime.session_store.create_session(
            "目标规则", provider="anthropic", model="claude-test", thinking_enabled=True
        )

        def fail(*args, **kwargs):
            raise sqlite3.OperationalError("模拟保存或加载失败")

        monkeypatch.setattr(runtime.session_store, failure_point, fail)
        with pytest.raises(sqlite3.OperationalError, match="模拟保存或加载失败"):
            runtime.resume_session(target.session_id)
        assert runtime.model is original_model
        assert runtime.session_id == original_id
        assert runtime.messages == original_messages
        assert runtime.last_result is original_result
        assert runtime.thinking_enabled
        assert runtime.session_info.thinking_enabled
        assert original_model.extra_body["thinking"]["type"] == "enabled"
        assert not original_model.client.closed
        assert models[-1].client.closed


def test_failed_toggle_does_not_change_request_or_memory_state(tmp_path, monkeypatch):
    model = OpenAICompatibleModel(model="deepseek-test", client=OfflineClient())
    with create_runtime(model=model, db_path=tmp_path / "sessions.sqlite3") as runtime:

        def fail(*args, **kwargs):
            raise sqlite3.OperationalError("模拟保存失败")

        monkeypatch.setattr(runtime.session_store, "update_thinking", fail)
        with pytest.raises(sqlite3.OperationalError, match="模拟保存失败"):
            runtime.set_thinking(True)
        assert not runtime.thinking_enabled
        assert not runtime.session_info.thinking_enabled
        assert model.extra_body["thinking"]["type"] == "disabled"


def test_cli_toggle_on_integrated_model_persists_and_updates_real_request_options(tmp_path):
    model = OpenAICompatibleModel(model="deepseek-test", client=OfflineClient())
    with create_runtime(model=model, db_path=tmp_path / "sessions.sqlite3") as runtime:
        console = CaptureConsole(["/think", "/exit"])
        cli.run(console=console, runtime=runtime)
        assert runtime.thinking_enabled
        assert runtime.session_info.thinking_enabled
        runtime.respond("确认实际开启")
        assert model.client.requests[-1]["extra_body"]["thinking"]["type"] == "enabled"
        assert "思考模式：开" in "\n".join(console.output)
