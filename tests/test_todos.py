import sqlite3

import pytest

from kairos.todos import add_todo, complete_todo, initialize_database, list_todos, update_todo


def test_initialize_database_creates_todos_table_with_expected_columns(tmp_path):
    connection = initialize_database(tmp_path / "kairos.sqlite3")

    columns = connection.execute("PRAGMA table_info(todos)").fetchall()

    assert [column[1] for column in columns] == [
        "id",
        "title",
        "due_at",
        "priority",
        "status",
        "project",
        "tags",
        "created_at",
        "updated_at",
    ]
    connection.close()


def test_initialize_database_is_idempotent_and_preserves_rows(tmp_path):
    db_path = tmp_path / "kairos.sqlite3"
    connection = initialize_database(db_path)
    connection.execute(
        """
        INSERT INTO todos (title, created_at, updated_at)
        VALUES (?, ?, ?)
        """,
        ("学习 SQLite", "2026-10-06T00:00:00+08:00", "2026-10-06T00:00:00+08:00"),
    )
    connection.commit()
    connection.close()

    reopened = initialize_database(db_path)

    row = reopened.execute("SELECT title FROM todos").fetchone()
    assert row["title"] == "学习 SQLite"
    reopened.close()


def test_todos_table_rejects_invalid_status(tmp_path):
    connection = initialize_database(tmp_path / "kairos.sqlite3")

    try:
        connection.execute(
            """
            INSERT INTO todos (title, status, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            ("错误状态", "unknown", "2026-10-06T00:00:00+08:00", "2026-10-06T00:00:00+08:00"),
        )
        connection.commit()
    except sqlite3.IntegrityError:
        pass
    else:
        raise AssertionError("无效的 status 不应写入 todos 表")
    finally:
        connection.close()


def test_add_todo_returns_saved_todo_with_defaults(tmp_path):
    connection = initialize_database(tmp_path / "kairos.sqlite3")

    todo = add_todo(connection, "学习 SQLite", tags=["Kairos", "学习"])

    assert todo["id"] == 1
    assert todo["title"] == "学习 SQLite"
    assert todo["priority"] == 2
    assert todo["status"] == "pending"
    assert todo["tags"] == ["Kairos", "学习"]
    assert todo["created_at"] == todo["updated_at"]
    connection.close()


def test_list_todos_can_filter_by_status_and_priority(tmp_path):
    connection = initialize_database(tmp_path / "kairos.sqlite3")
    add_todo(connection, "普通待办")
    high_priority = add_todo(connection, "高优先级待办", priority=3)
    complete_todo(connection, high_priority["id"])

    pending = list_todos(connection, status="pending")
    completed_high = list_todos(connection, status="completed", priority=3)

    assert [todo["title"] for todo in pending] == ["普通待办"]
    assert [todo["title"] for todo in completed_high] == ["高优先级待办"]
    connection.close()


def test_update_todo_changes_only_requested_fields(tmp_path):
    connection = initialize_database(tmp_path / "kairos.sqlite3")
    todo = add_todo(connection, "原始标题", project="Kairos")

    updated = update_todo(connection, todo["id"], title="新标题", due_at="2026-10-10T17:00:00+08:00")

    assert updated["title"] == "新标题"
    assert updated["due_at"] == "2026-10-10T17:00:00+08:00"
    assert updated["project"] == "Kairos"
    connection.close()


def test_todo_functions_reject_invalid_input(tmp_path):
    connection = initialize_database(tmp_path / "kairos.sqlite3")

    with pytest.raises(ValueError, match="标题"):
        add_todo(connection, "   ")
    with pytest.raises(ValueError, match="priority"):
        add_todo(connection, "错误优先级", priority=4)
    with pytest.raises(KeyError, match="找不到待办"):
        complete_todo(connection, 999)
    with pytest.raises(ValueError, match="不支持修改"):
        update_todo(connection, 1, unsupported="value")
    connection.close()
