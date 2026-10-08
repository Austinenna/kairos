import json

from kairos.observability import JsonlLogger


def test_jsonl_logger_appends_timestamped_json_event(tmp_path):
    log_path = tmp_path / "logs" / "agent.jsonl"
    logger = JsonlLogger(log_path)

    logger({"event": "test", "value": 1})
    logger({"event": "second", "value": "中文"})

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["event"] == "test"
    assert first["value"] == 1
    assert first["timestamp"].endswith("+00:00")
    assert second["event"] == "second"
    assert second["value"] == "中文"
