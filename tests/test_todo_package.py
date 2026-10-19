import sqlite3
import subprocess
import sys
import textwrap
from contextlib import closing
from pathlib import Path

import pytest

from kairos_todo import initialize_database, initialize_schema
from kairos_todo.tools import build_tools


def test_package_runs_with_only_standard_library_and_persists_crud(tmp_path):
    source_root = Path(__file__).resolve().parents[1] / "src"
    script = textwrap.dedent(
        """
        import sqlite3
        import sys
        from contextlib import closing

        sys.path.insert(0, sys.argv[1])
        from kairos_todo import (
            add_todo, complete_todo, delete_todo, initialize_database,
            list_todos, update_todo,
        )
        from kairos_todo.tools import build_tools

        path = sys.argv[2]
        with closing(initialize_database(path)) as connection:
            assert connection.row_factory is sqlite3.Row
            first = add_todo(connection, "记录事项", tags=["独立包"])
            updated = update_todo(connection, first["id"], title="修改事项", priority=3)
            assert updated["title"] == "修改事项"
            completed = complete_todo(connection, first["id"])
            assert completed["status"] == "completed"
            temporary = add_todo(connection, "待删除")
            assert delete_todo(connection, temporary["id"]) == temporary
            assert list_todos(connection) == [completed]

        with closing(initialize_database(path)) as reopened:
            assert list_todos(reopened, status="completed", priority=3) == [completed]
            bundle = build_tools(reopened)
            assert type(bundle) is dict
            assert bundle["handlers"]["list_todos"]() == [completed]
            delete_todo(reopened, completed["id"])

        with closing(initialize_database(path)) as reopened:
            assert list_todos(reopened) == []

        assert not any(name == "kairos" or name.startswith("kairos.") for name in sys.modules)
        """
    )
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(source_root), str(tmp_path / "todos.sqlite3")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_initialize_schema_only_creates_todo_structure_and_is_idempotent(tmp_path):
    with closing(sqlite3.connect(tmp_path / "todos.sqlite3")) as connection:
        connection.row_factory = sqlite3.Row
        initialize_schema(connection)
        initialize_schema(connection)

        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        indexes = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'todos' ORDER BY name"
        ).fetchall()
        assert [row["name"] for row in tables] == ["todos"]
        assert [row["name"] for row in indexes] == ["idx_todos_due_at", "idx_todos_status"]
        assert not connection.in_transaction


def test_initialize_schema_preserves_other_tables_and_data(tmp_path):
    with closing(sqlite3.connect(tmp_path / "shared.sqlite3")) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("CREATE TABLE unrelated (value TEXT)")
        connection.execute("INSERT INTO unrelated VALUES ('preserved')")
        connection.commit()

        initialize_schema(connection)

        assert connection.execute("SELECT value FROM unrelated").fetchone()["value"] == "preserved"
        assert connection.row_factory is sqlite3.Row


def test_initialize_schema_rejects_active_transaction_without_committing_it(tmp_path):
    path = tmp_path / "shared.sqlite3"
    with closing(sqlite3.connect(path)) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("CREATE TABLE unrelated (value TEXT)")
        connection.execute("INSERT INTO unrelated VALUES ('uncommitted')")

        with pytest.raises(ValueError, match="当前事务"):
            initialize_schema(connection)

        assert connection.in_transaction
        assert connection.execute("SELECT COUNT(*) FROM unrelated").fetchone()[0] == 1
        assert connection.execute("SELECT name FROM sqlite_master WHERE name = 'todos'").fetchone() is None
        with closing(sqlite3.connect(path)) as observer:
            assert observer.execute("SELECT COUNT(*) FROM unrelated").fetchone()[0] == 0
        connection.rollback()


def test_initialize_schema_requires_row_factory_without_changing_connection(tmp_path):
    with closing(sqlite3.connect(tmp_path / "todos.sqlite3")) as connection:
        with pytest.raises(ValueError, match="sqlite3.Row"):
            initialize_schema(connection)

        assert connection.row_factory is None
        assert connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall() == []


def test_build_tools_creates_storage_and_returns_independent_definitions(tmp_path):
    with closing(sqlite3.connect(tmp_path / "todos.sqlite3")) as connection:
        connection.row_factory = sqlite3.Row
        bundle = build_tools(connection)
        handlers = bundle["handlers"]
        assert set(bundle) == {"handlers", "schemas", "instructions"}
        assert set(handlers) == {"add_todo", "list_todos", "update_todo", "complete_todo", "delete_todo"}
        assert {schema["function"]["name"] for schema in bundle["schemas"]} == set(handlers)
        assert isinstance(bundle["instructions"], tuple)
        assert "明确要求删除" in "\n".join(bundle["instructions"])

        added = handlers["add_todo"](title="通过普通工具集合调用")
        handlers["update_todo"](todo_id=added["id"], priority=3)
        completed = handlers["complete_todo"](todo_id=added["id"])
        assert handlers["list_todos"](status="completed") == [completed]
        assert handlers["delete_todo"](todo_id=added["id"]) == completed
        assert handlers["list_todos"]() == []

        bundle["schemas"][0]["function"]["name"] = "changed"
        assert build_tools(connection)["schemas"][0]["function"]["name"] == "add_todo"


def test_bound_tool_rolls_back_partial_database_write_on_failure(tmp_path):
    with closing(initialize_database(tmp_path / "todos.sqlite3")) as connection:
        bundle = build_tools(connection)
        connection.executescript(
            """
            CREATE TABLE audit (title TEXT);
            CREATE TRIGGER fail_todo_insert BEFORE INSERT ON todos
            BEGIN
                INSERT INTO audit VALUES (NEW.title);
                SELECT RAISE(FAIL, 'synthetic failure');
            END;
            """
        )

        with pytest.raises(sqlite3.IntegrityError, match="synthetic failure"):
            bundle["handlers"]["add_todo"](title="不能留下半次写入")

        assert not connection.in_transaction
        assert connection.execute("SELECT * FROM audit").fetchall() == []
        assert bundle["handlers"]["list_todos"]() == []
