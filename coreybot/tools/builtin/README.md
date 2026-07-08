# 内置工具开发规范（强制）

本目录（`coreybot/tools/builtin/`）下的每个内置工具都**必须**遵循以下规范。
这些规则不是建议，而是由 `tests/test_builtin_conventions.py` 中的**一致性测试**强制
校验的：任何不符合的工具都会让 `pytest` 失败（相当于项目自带的 lint 规则）。

## 目录结构

每个工具是一个**独立子包目录** `coreybot/tools/builtin/<name>/`，包含：

```text
<name>/
  __init__.py        # 重导出：SPEC 与工具函数
  spec.py            # 接口声明（ToolSpec：name/description/parameters）
  tool.py            # 实现，用 @tool(spec=SPEC) 注册
  tests/
    __init__.py
    test_<name>.py   # 与实现同目录的单元测试
```

## 强制规则

1. **必须有 `spec.py`**，且其中定义一个模块级变量 `SPEC`，类型为 `ToolSpec`。
2. **`spec.py` 只放接口声明**：不得包含函数/类定义或分支逻辑（`def` / `class` /
   `if` / `for` / `while` / `try`）。它只描述工具对模型暴露的契约。
3. **必须有 `tool.py`**，其中用 `@tool(spec=SPEC)` 注册实现（从 `.spec` 导入 `SPEC`）。
   不要在内置工具里用内联的 `@tool(name=..., description=...)` 写法。
4. **`__init__.py` 必须重导出 `SPEC`**（`from .spec import SPEC`），方便集中查看接口。
5. **必须有 `tests/` 目录**，且至少包含一个 `test_*.py`。
6. **注册名一致**：`SPEC.name` 必须与它在默认注册表中的键一致；不同工具的 `SPEC.name`
   不得重复。
7. **参数提示格式**：`SPEC.parameters` 的每个值建议写成 `"<类型> -- <说明>"`。

> 自动发现：`builtin/__init__.py` 会导入任何带 `tool.py` 的子包，因此**新增工具无需**
> 改任何中心文件——建目录、写 `spec.py` + `tool.py` + `tests/` 即可。

## 最小示例

```python
# echo/spec.py
from coreybot.tools import ToolSpec

SPEC = ToolSpec(
    name="echo",
    description="Echo the input text back.",
    parameters={"text": "string -- the text to echo"},
)
```

```python
# echo/tool.py
from coreybot.tools import tool, ToolResult
from .spec import SPEC

@tool(spec=SPEC)
def echo(text: str) -> ToolResult:
    return ToolResult.success(text)
```

```python
# echo/__init__.py
from .spec import SPEC
from .tool import echo

__all__ = ["SPEC", "echo"]
```

## 内置高危工具与安全标注

有副作用的工具必须在 `spec.py` 的 `SPEC` 上声明 `safety=make_profile(...)`，这样 `SafetyPolicy` 才能在执行**前**对调用做可恢复性分级（FULL / PARTIAL / NONE）。未声明 `safety` 的工具默认为不透明，会被保守地归为 NONE。

当前内置的高危工具：

- **`write_file`**（`FS_WRITE`）：工作区内写入判为 **FULL**，执行前快照原文、可字节级回滚；目标落在工作区之外或 `~/.coreybot` 内则降为 NONE。
- **`bash`**（`EXEC` + `DESTRUCTIVE`）：执行任意 shell 命令。命令命中危险模式（如 `rm -rf`），或在无审批回调的无头环境中，一律判为 **NONE** 并拒绝（YOLO 下唯一的人工闸门）。需要放宽时，给 `SafetyPolicy` 注入 approval handler 即可。
- **`webtool`**（`NETWORK` + `EXTERNAL_SIDE_EFFECT` + compensation）：HTTP(S) 请求。因为请求可能改变远端状态且本地无法回滚，判为 **PARTIAL**：不阻断执行，但会把 compensation 备注记入审计。

## 如何本地校验

```powershell
# 只跑规范一致性检查
.\.venv\Scripts\python.exe -m pytest coreybot\tools\builtin\tests\test_builtin_conventions.py
# 或跑某个工具的全部测试
.\.venv\Scripts\python.exe -m pytest coreybot\tools\builtin\calc
```

