import json

from kairos.tool_schemas import TODO_TOOL_SCHEMAS


def test_todo_tool_schemas_have_expected_names_and_json_shape():
    names = [tool["function"]["name"] for tool in TODO_TOOL_SCHEMAS]

    assert names == ["add_todo", "list_todos", "update_todo", "complete_todo"]
    assert all(tool["type"] == "function" for tool in TODO_TOOL_SCHEMAS)
    assert all(tool["function"]["parameters"]["type"] == "object" for tool in TODO_TOOL_SCHEMAS)
    assert all(tool["function"]["parameters"]["additionalProperties"] is False for tool in TODO_TOOL_SCHEMAS)

    json.dumps(TODO_TOOL_SCHEMAS, ensure_ascii=False)


def test_add_todo_schema_requires_title_and_describes_priority():
    parameters = TODO_TOOL_SCHEMAS[0]["function"]["parameters"]

    assert parameters["required"] == ["title"]
    assert parameters["properties"]["priority"]["enum"] == [1, 2, 3]
    assert parameters["properties"]["tags"]["items"] == {"type": "string"}


def test_list_todos_schema_has_only_optional_filters():
    parameters = TODO_TOOL_SCHEMAS[1]["function"]["parameters"]

    assert parameters["required"] == []
    assert set(parameters["properties"]) == {"status", "priority"}


def test_update_and_complete_schemas_require_todo_id():
    update_parameters = TODO_TOOL_SCHEMAS[2]["function"]["parameters"]
    complete_parameters = TODO_TOOL_SCHEMAS[3]["function"]["parameters"]

    assert update_parameters["required"] == ["todo_id"]
    assert update_parameters["minProperties"] == 2
    assert complete_parameters["required"] == ["todo_id"]
