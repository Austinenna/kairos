from contextlib import closing

import pytest

from kairos.todos import add_todo, complete_todo, delete_todo, initialize_database, list_todos, update_todo


@pytest.mark.parametrize("status", ["pending", "completed"])
def test_delete_todo_returns_record_and_only_removes_target(tmp_path, status):
    with closing(initialize_database(tmp_path / "kairos.sqlite3")) as connection:
        retained = add_todo(connection, "保留这条待办", tags=["保留"])
        target = add_todo(connection, "删除这条待办", status=status, tags=["工作", "测试"])

        deleted = delete_todo(connection, target["id"])

        assert deleted == target
        assert deleted["tags"] == ["工作", "测试"]
        assert list_todos(connection) == [retained]
        assert connection.execute("SELECT * FROM todos WHERE id = ?", (target["id"],)).fetchone() is None


def test_delete_todo_persists_when_database_reopens(tmp_path):
    db_path = tmp_path / "kairos.sqlite3"
    with closing(initialize_database(db_path)) as connection:
        target = add_todo(connection, "持久化删除")
        retained = add_todo(connection, "仍然保留")
        delete_todo(connection, target["id"])

    with closing(initialize_database(db_path)) as connection:
        assert list_todos(connection) == [retained]


def test_delete_todo_rejects_missing_and_already_deleted_ids(tmp_path):
    with closing(initialize_database(tmp_path / "kairos.sqlite3")) as connection:
        retained = add_todo(connection, "保留这条待办")
        target = add_todo(connection, "只能删除一次")
        delete_todo(connection, target["id"])

        for todo_id in (target["id"], target["id"] + 100):
            with pytest.raises(KeyError, match=f"找不到待办：{todo_id}"):
                delete_todo(connection, todo_id)
            assert list_todos(connection) == [retained]


@pytest.mark.parametrize("todo_id", [True, False, 1.0, "1", 0, -1, None])
def test_invalid_ids_cannot_mutate_existing_todo(tmp_path, todo_id):
    with closing(initialize_database(tmp_path / "kairos.sqlite3")) as connection:
        retained = add_todo(connection, "不能误操作 ID 1")
        for operation in (
            lambda: delete_todo(connection, todo_id),
            lambda: complete_todo(connection, todo_id),
            lambda: update_todo(connection, todo_id, title="不应修改"),
        ):
            with pytest.raises(ValueError, match="todo_id"):
                operation()
            assert list_todos(connection) == [retained]
            assert not connection.in_transaction


@pytest.mark.parametrize(
    "changes",
    [
        {"title": " "},
        {"priority": True},
        {"priority": 2.0},
        {"priority": "2"},
        {"tags": "工作"},
        {"tags": {"工作": True}},
        {"tags": ["工作", 1]},
        {"tags": [" "]},
        {"due_at": "明天下午"},
        {"due_at": "2026-10-18"},
        {"due_at": "2026-10-18T15:00:00"},
        {"due_at": "2026-02-30T15:00:00+08:00"},
        {"due_at": "2026-10-18T15:00:00+00:60"},
        {"due_at": 123},
        {"due_at": {"invalid": "object"}},
        {"project": True},
        {"status": "invalid"},
    ],
)
def test_invalid_fields_leave_database_unchanged_on_add_and_update(tmp_path, changes):
    with closing(initialize_database(tmp_path / "kairos.sqlite3")) as connection:
        retained = add_todo(connection, "原始待办")
        with pytest.raises(ValueError):
            add_todo(connection, **{"title": "不应写入", **changes})
        with pytest.raises(ValueError):
            update_todo(connection, retained["id"], **changes)
        assert list_todos(connection) == [retained]
        assert not connection.in_transaction


def test_valid_fields_round_trip_and_optional_fields_can_be_cleared(tmp_path):
    with closing(initialize_database(tmp_path / "kairos.sqlite3")) as connection:
        todo = add_todo(
            connection,
            " 合法待办 ",
            due_at="2026-10-18T15:00:00+08:00",
            priority=3,
            project="Kairos",
            tags=[" 工作 ", "测试"],
        )
        assert todo["tags"] == ["工作", "测试"]
        assert todo["due_at"] == "2026-10-18T15:00:00+08:00"
        updated = update_todo(connection, todo["id"], due_at="2026-10-18T07:00:00Z")
        assert updated["due_at"] == "2026-10-18T07:00:00+00:00"
        cleared = update_todo(connection, todo["id"], due_at=None, project=None, tags=[])
        assert (cleared["due_at"], cleared["project"], cleared["tags"]) == (None, None, [])
