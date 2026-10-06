import sqlite3

from kairos.todos import initialize_database


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
