from contextlib import closing

import pytest

from kairos.todos import add_todo, delete_todo, initialize_database, list_todos


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
