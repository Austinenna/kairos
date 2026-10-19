"""待办工具的 JSON Schema。

Schema 只描述模型可以调用什么工具、工具需要哪些参数；
真正执行数据库操作的函数在 ``kairos_todo.storage`` 中。
"""

ADD_TODO_TOOL = {
    "type": "function",
    "function": {
        "name": "add_todo",
        "description": "新增一条待办。默认优先级为普通，默认状态为未完成。",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "待办标题，不能为空。"},
                "due_at": {
                    "type": ["string", "null"],
                    "format": "date-time",
                    "description": "截止时间，使用含时分秒和时区的 ISO 8601 格式；null 表示无截止时间。",
                },
                "priority": {
                    "type": "integer",
                    "enum": [1, 2, 3],
                    "description": "优先级：1 为低，2 为普通，3 为高。",
                },
                "status": {
                    "type": "string",
                    "enum": ["pending", "completed"],
                    "description": "待办状态，默认为 pending。",
                },
                "project": {"type": ["string", "null"], "description": "所属项目，可选。"},
                "tags": {
                    "type": ["array", "null"],
                    "items": {"type": "string", "minLength": 1},
                    "description": "标签列表，可选。",
                },
            },
            "required": ["title"],
            "additionalProperties": False,
        },
    },
}

LIST_TODOS_TOOL = {
    "type": "function",
    "function": {
        "name": "list_todos",
        "description": "查询待办，可按状态或优先级筛选。",
        "parameters": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["pending", "completed"],
                    "description": "只返回指定状态的待办，可选。",
                },
                "priority": {
                    "type": "integer",
                    "enum": [1, 2, 3],
                    "description": "只返回指定优先级的待办，可选。",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
}

UPDATE_TODO_TOOL = {
    "type": "function",
    "function": {
        "name": "update_todo",
        "description": "修改一条待办中明确提供的字段，至少提供一个要修改的字段。",
        "parameters": {
            "type": "object",
            "properties": {
                "todo_id": {"type": "integer", "minimum": 1, "description": "要修改的待办 ID。"},
                "title": {"type": "string", "description": "新的待办标题。"},
                "due_at": {
                    "type": ["string", "null"],
                    "format": "date-time",
                    "description": "新的截止时间，使用含时分秒和时区的 ISO 8601 格式；null 表示无截止时间。",
                },
                "priority": {
                    "type": "integer",
                    "enum": [1, 2, 3],
                    "description": "新的优先级：1 为低，2 为普通，3 为高。",
                },
                "status": {
                    "type": "string",
                    "enum": ["pending", "completed"],
                    "description": "新的待办状态。",
                },
                "project": {"type": ["string", "null"], "description": "新的所属项目。"},
                "tags": {
                    "type": ["array", "null"],
                    "items": {"type": "string", "minLength": 1},
                    "description": "新的标签列表。",
                },
            },
            "required": ["todo_id"],
            "minProperties": 2,
            "additionalProperties": False,
        },
    },
}

COMPLETE_TODO_TOOL = {
    "type": "function",
    "function": {
        "name": "complete_todo",
        "description": "把指定的待办标记为已完成。",
        "parameters": {
            "type": "object",
            "properties": {
                "todo_id": {"type": "integer", "minimum": 1, "description": "要完成的待办 ID。"},
            },
            "required": ["todo_id"],
            "additionalProperties": False,
        },
    },
}

DELETE_TODO_TOOL = {
    "type": "function",
    "function": {
        "name": "delete_todo",
        "description": "按 ID 永久删除一条待办并返回被删除的记录。仅在用户要求删除且目标明确时调用。",
        "parameters": {
            "type": "object",
            "properties": {
                "todo_id": {"type": "integer", "minimum": 1, "description": "要删除的待办 ID。"},
            },
            "required": ["todo_id"],
            "additionalProperties": False,
        },
    },
}

TODO_TOOL_SCHEMAS = [
    ADD_TODO_TOOL,
    LIST_TODOS_TOOL,
    UPDATE_TODO_TOOL,
    COMPLETE_TODO_TOOL,
    DELETE_TODO_TOOL,
]
