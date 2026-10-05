# Kairos 项目协作约定

- 接手项目时先阅读 `README.md`、`HANDOFF.md` 和 `CLAUDE.md`。
- `docs/archive/`（如果后续建立）只存已废弃方案，不作为当前实现要求。
- 核心 agent 循环、模型适配层和记忆层保持手写；实验结论写入 `docs/learning-log.md`。
- 使用项目根目录的 `.env` 读取密钥，`data/` 仅存本地个人数据；两者不得进入 Git。
- 运行、测试和格式化优先使用 `uv`，修改后先完成必要验证，再提交独立的本地 Git 改动。
