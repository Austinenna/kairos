"""定义主 Agent 与后台提取器分别可用的记忆工具。"""

REMEMBER_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "remember",
        "description": (
            "仅在用户明确要求记住时，将内容追加到每次加载的 CORE。"
            "先检查已有核心记忆，重复内容不再保存；本期不覆盖冲突、修改旧事实或遗忘。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "minLength": 1, "description": "用户明确要求保存的内容。"},
            },
            "required": ["content"],
            "additionalProperties": False,
        },
    },
}

READ_MEMORY_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_memory",
        "description": "按完整记忆索引中的主题 ID 读取正文、用途和来源；不能传文件路径。",
        "parameters": {
            "type": "object",
            "properties": {
                "topic_id": {"type": "string", "description": "MEMORY 索引列出的稳定主题 ID。"},
            },
            "required": ["topic_id"],
            "additionalProperties": False,
        },
    },
}

WRITE_CORE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "write_core",
        "description": (
            "只供后台提取器将用户本人直接表达的姓名/称呼、通用交流偏好和通用协作约定追加到 CORE，"
            "无需用户说记住。先检查 CORE，已覆盖则跳过；仅 topics 已有时仍可提升并保留原主题。"
            "普通职业、学习和项目背景写 topics；不覆盖冲突、不修改旧事实或删除。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "minLength": 1,
                    "description": "符合自动核心范围、由用户直接表达且 CORE 尚未覆盖的新增内容。",
                },
            },
            "required": ["content"],
            "additionalProperties": False,
        },
    },
}


WRITE_TOPIC_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "write_topic",
        "description": (
            "只供后台提取器创建主题或追加新增信息。先读取可能相关的已有主题查重；"
            "一个文件表达一个主题，不创建分类、不覆盖旧正文、不修改 CORE，索引由程序生成。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic_id": {
                    "type": ["string", "null"],
                    "description": "追加时使用已有主题 ID；创建新主题时为 null。",
                },
                "title": {
                    "type": ["string", "null"],
                    "description": "创建时必填的简短主题标题；追加时为 null，不修改原标题。",
                },
                "description": {
                    "type": ["string", "null"],
                    "description": "创建时必填的读取用途；追加时为 null，不修改原用途。",
                },
                "addition": {"type": "string", "minLength": 1, "description": "仅包含尚未保存的新增正文。"},
            },
            "required": ["topic_id", "title", "description", "addition"],
            "additionalProperties": False,
        },
    },
}

PRIMARY_MEMORY_TOOL_SCHEMAS = [REMEMBER_TOOL_SCHEMA, READ_MEMORY_TOOL_SCHEMA]
EXTRACTION_MEMORY_TOOL_SCHEMAS = [READ_MEMORY_TOOL_SCHEMA, WRITE_CORE_TOOL_SCHEMA, WRITE_TOPIC_TOOL_SCHEMA]
