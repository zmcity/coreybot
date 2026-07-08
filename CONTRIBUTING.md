# 贡献指南

感谢你对 **coreybot** 的兴趣！这是一个从零开始、极少依赖的异步 LLM agent 学习项目。

## 开发环境

项目基于 **Python 3.11+**，推荐使用虚拟环境：

```bash
python -m venv .venv
# Windows: .\\.venv\\Scripts\\activate    Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
```

## 运行测试

```bash
python -m pytest -q
```

提交 PR 前请确保全部测试通过。若新增功能，请一并补充对应的单元 / 集成测试。

## 代码风格

- 代码标识符与 docstring 用**英文**；聊天 / 文档用**中文**。
- 保持改动最小、与现有风格一致。
- 文件统一 UTF-8（无 BOM）、LF 换行（由 `.gitattributes` 约束）。
- 内置工具需遵循 `coreybot/tools/builtin/` 的约定（每个工具一个目录 + `spec.py` + `tool.py` + `tests/`）。

## 分支与提交

- `main` 为受保护分支，请从特性分支发起 Pull Request。
- 提交信息建议遵循 [Conventional Commits](https://www.conventionalcommits.org/)，例如 `feat:` / `fix:` / `chore:` / `docs:`。

