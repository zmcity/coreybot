# coreybot

一个从零搭建、尽量少依赖的 **LLM Agent** 学习项目。Agent 核心（消息模型、多协议兼容层、聊天循环）尽量只用 Python 标准库，让逻辑清晰易读；唯一复用成熟库的地方是终端界面（[Textual](https://textual.textualize.io/)），因为「渲染」属于呈现层，而非你要学习的 Agent 核心。

## 功能特性

- **统一消息模型**：所有协议共用同一套中立数据结构。
- **可扩展的兼容层**：基于注册表，新增协议只需写一个子类并用 `@register("name")` 注册。
- **内置三种协议**：OpenAI 兼容（默认）、Anthropic、Gemini。
- **流式输出**：基于 SSE，在 TUI 中呈现打字机效果。
- **两种前端**：类 Codex 的 Textual **TUI**，以及纯命令行 **CLI** 循环。
- **异步架构 + 会话中断**：整条通信链路（HTTP → provider → agent → 前端）都是 `async` 的，并贯穿一个类 Go `context` 的 `CancelToken`；TUI 里按 `Esc`、CLI 里按 `Ctrl+C` 即可中断正在进行的请求。
- **Markdown 富文本渲染**：助手回复按 Markdown 渲染，行内代码 `code`、```大代码块``` 与「> 引用」都会高亮显示；并明确禁止 LaTeX / `$...$`，避免终端里出现 `$` 乱码。
- **分屏 TUI + 实时流程图**：左侧紧凑聊天，右侧用树状流程图实时展示 llm / tool 等活动，且**为未来的 MCP / skill / 声明式 agent 预留了可扩展位**。
- **完善的测试**：大量单元测试 + 集成测试（`pytest`），运行 `pytest` 即可全绿。

## 交互协议（XML 标签）

模型每轮的输出被约束为 **XML 标签** 格式，而非自由文本。相比 JSON，标签边界更抡噪，且标签内的文本无需转义（引号、换行、代码、`&` 等都可直接写），更适合后续的工具调用。

当前支持的格式：

```xml
<message>给用户的回复</message>
```

说明：

- 系统提示词由 `protocol.build_system_prompt()` 自动拼接协议说明。
- `protocol.parse_agent_response()` 使用自写的**宽松标签解析器**（大小写不敏感、允许跨行、容忍标签前后杂散文本）。
- 若模型未输出 `<message>` 标签，会**优雅降级**：把原文当作回复展示，并附一条协议警告，不会崩溃。
- 工具调用使用 `<tool_call>` 变体，详见下方「工具系统」章节。

### 回复正文用 Markdown

`<message>` 内的正文被约定为 **GitHub 风格 Markdown**，TUI 会用 Rich 渲染它，方便浏览代码与引用：

- 行内代码：用单反引号，如 `variable`。
- 代码块：用三反引号并带语言，如 ```python …```，会按语言高亮。
- 引用 / 提示：以 `> ` 开头。
- 也支持 **粗体**、*斜体*、列表、标题。
- **不要**使用 LaTeX 或 `$...$` 公式：终端不渲染数学，会原样显示为美元符号。需要数学时请用纯文本或代码块表示。

这条约定写在系统提示词里（见 `coreybot/llm/protocol.py` 的 `PROTOCOL_INSTRUCTIONS`），因此对所有 provider 生效。注意：只有「Markdown 正文里的代码」才用代码围栏；`<message>` / `<tool_call>` 这些**标签本身**不要放进代码围栏。

> 说明：TUI 与 CLI（`--cli`）都会用 Rich 渲染助手回复的 Markdown。CLI 在真实终端里输出彩色，在管道 / 重定向 / 测试等非 TTY 场景下自动降级为纯文本；若未安装 Rich 也会回退为原始文本。

## 工具系统（Tool Calling）

Agent 不仅能聊天，还能调用工具。每个用户回合内部会运行一个小循环：**模型 → （可选）调用工具 → 观察结果 → 继续 → 最终回复**，并有 `max_steps` 上限防止无限循环。

### 内置工具

- `calc(expression, variables?, mode?)` — 代数/科学计算器（基于 SymPy）：四则运算、化简/因式分解、微积分、解方程；支持 `^` 幂、隐式乘法、`pi/e/i` 与 `sin/sqrt/log` 等函数。解析前做属性访问/dunder 白名单校验并中和 `__builtins__`，防止任意代码执行。
- `current_time()` — 返回当前本地时间（ISO-8601）。
- `read_file(path, max_bytes?)` — 读取 UTF-8 文本文件（有大小上限）。

### 定义你自己的工具

每个内置工具是一个**独立子包目录**，并且把「接口声明」「实现」「测试」拆成三块，方便单独观察与自动化测试：

```text
coreybot/tools/builtin/
  calc/
    __init__.py          # 重导出：from .tool import calc; from .spec import SPEC
    spec.py              # 接口声明（ToolSpec：name / description / parameters）
    tool.py              # 实现，用 @tool(spec=SPEC) 注册
    tests/
      test_calc.py       # 与工具同目录的单元测试
  clock/
  read_file/
```

**接口声明**单独放在 `spec.py`，一眼就能看清工具对模型暴露的契约：

```python
# calc/spec.py
from coreybot.tools import ToolSpec

SPEC = ToolSpec(
    name="calc",
    description="Evaluate a basic arithmetic expression, e.g. '2 * (3 + 4)'.",
    parameters={"expression": "string -- the arithmetic expression to evaluate"},
)
```

**实现**在 `tool.py` 里引用该声明并注册（逻辑与契约分离）：

```python
# calc/tool.py
from coreybot.tools import tool, ToolResult
from .spec import SPEC

@tool(spec=SPEC)
def calc(expression: str) -> ToolResult:
    ...
```

> 也支持把声明写在装饰器里的内联写法：`@tool(name=..., description=..., parameters=...)`；二者等价，`spec=` 版更适合需要「集中查看接口」的场景。

新增一个工具只需**新建一个目录** `coreybot/tools/builtin/<名字>/`，放 `spec.py` + `tool.py`（用 `@tool(spec=SPEC)` 注册）和一个 `tests/` 即可——`builtin/__init__.py` 会**自动发现**任何带 `tool.py` 的子包并导入它（无需改任何中心文件）。在界面里用 `/tools` 可查看当前已注册的工具。

调用协议（模型输出）：

```xml
<tool_call>
  <name>calc</name>
  <arguments>{"expression": "2 * (3 + 4)"}</arguments>
</tool_call>
```

`<arguments>` 采用内嵌 JSON 以保留参数类型（数字/布尔/数组）。未知工具、参数错误、工具异常都会被捕获并反馈给模型，而不会让程序崩溃。

## 分屏界面与实时流程图

TUI 采用**分屏**布局，模仿 Codex 的观感，让你在聊天的同时「看见」Agent 的内部执行过程：

```text
┌───────────────────────────┬───────────────────────┐
│ 聊天区（仅人类对话）        │ 流程图（整幅可拖动）     │
│ you │ 用 calc 算 21*2     │  ╭──────────────╮        │
│ bot │ 21 × 2 = **42**     │  │ 🧑 you  21*2 │        │
│                           │  ╰──────┬───────╯        │
│ （工具 / 通知 / 系统信息  │         ▼                │
│   全部移到右侧流程图）    │  ╭──────────────╮        │
│                           │  │ 🔧 calc  ✓ 42│        │
│                           │  ╰──────┬───────╯        │
│                           │         ▼   ←自动滚到最新 │
│                           │  ╭──────────────╮        │
│                           │  │ ✅ answer     │        │
│                           │  ╰──────────────╯        │
└───────────────────────────┴───────────────────────┘
```

- **左侧聊天区**：无边框、零外边距的紧凑气泡，用彩色角色前缀（`you` / `bot`）+ `│` 分隔正文，读起来像真正的聊天客户端。**聊天区只留人类对话**（你的提问 + 模型的最终回复）：工具调用、运行通知、启动横幅、中断提示等**系统 / 调试信息一律以 `notice` 事件形式进入右侧流程图**，不再淡化对话（`/help` `/tools` `/history` 等你主动请求的文本仍显示在左侧）。
- **右侧流程图**：不再是静态树，而是一张**真正的节点图**——每个执行步骤是一个**独立的方框控件**，方框之间用连线（带箭头的直角连接线）相连，随执行**实时生长**（用户 → 模型 → 工具 → 模型 → 回复）。节点带**状态样式**（进行中 = 黄框 / 成功 = 绿框 / 失败 = 红框）与结果摘要。**整个会话的所有回合都会留在图上**（由 `turn_start` / `turn_end` 划分，新回合从自己的用户节点另起一枝），不会因为你发了下一句而被清空。
- **整幅可拖动 + 自动跟随**：用鼠标**按住背景拖动**即可平移整个画布（grab-to-pan）；而当一个回合正在执行时，视图会**自动滚动到最新节点**（像日志窗口）。一旦你手动拖动，自动跟随就会暂停；下一个回合开始时又会恢复。实现上用了 Textual 的鼠标事件（`MouseDown` / `MouseMove` / `MouseUp` + `capture_mouse`）去调整滚动偏移（`scroll_to`），节点用绝对定位（`position: absolute` + `offset`）摆放，连线由一个背后图层用 `render_line` 直接绘制。画布**不显示滚动条**（`scrollbar-size: 0 0`），移动完全靠**自然拖拽**——而不是拖滚动条（`overflow: auto` 仍保留以便 `scroll_to` 可用于自动跟随）。为了让画布**在任何方向都能自由拖动（像地图）**，虚拟画布总是比可视区**在两个轴上都多出一圈留白**（`_CANVAS_PAD`，按 `内容 + 边距` 与 `可视区 + 边距` 取较大值）：即使图很小或比面板还窄，也有横纵双向的拖动余量（之前因为虚拟区正好等于内容导致`max_scroll` 为 0、拖不动）。你也可以**直接按住节点拖动**整幅画布（拖动与“点击展开/折叠”靠位移阈值区分）。拖到**边界**时，平移会**自己先把目标偏移量限制在 `[0, max_scroll]`（`_clamp`）**，并用 `scroll_to(..., animate=False, immediate=True)` **立即应用**，因此画布会严格跟随光标 1:1、松手后立即停住；而不会像之前那样 —— 因为 `scroll_to` 默认是**延迟 + 缓动**的，越拖过头时会排队一个被截断到最大值的目标，导致松手后**还会继续滑到边界（漂移）**。
- **遵循“遥测投影”架构**：流程图不再自己维护回合状态，而是 Agent 追加式遥测日志（`Agent.telemetry`，见 `coreybot/runtime/agent.py`）的**纯投影**：`set_history(events)` 根据完整日志整图重建（幂等），`append(event)` 则是单事件的增量快路。这把**渲染与聊天循环解耦**：聊天循环只负责产生事件，流程图只负责把事件画出来，以后新增特性（比如另一种可视化）只需读同一份遥测，互不影响。
- **正在运行的步骤会呼吸 + 实时计时**：正在执行的节点（llm / tool）会让**边框柔和地呼吸**（**边框颜色代表节点类型**：llm / tool / mcp / skill / agent 各自一种颜色（分别为 `$primary` / `$success` / `$secondary` / `$warning` / `$accent`）；边框始终是圆角，呼吸时只向**同色系更亮的强调色（`$*-lighten-3`）**轻轻过渡，而不是换成琥珀色这样的另一种颜色，约 1.1Hz，不再是硬闪 / 加粗；失败的节点则无论类型都显示红色边框），把你的注意力引到**当前在跑的那一步**；同时头部带一个**计时器**：**2 秒以内用毫秒（如 `840ms`）、超过 2 秒用秒（如 `3.4s`，≥10s 去掉小数位如 `12s`）**。正在运行的任务**实时计时**（毫秒模式下最快每 100ms 刷新一次，避免影响性能），完成后计时**冻结**在最终耗时；若这一步被中断或回合结束，正在跑的节点也会立即冻结（而不会一直往上计数）。实现上只用**一个 10fps 定时器**且**仅在有节点运行时开启**，每帧只重绘正在运行的方框（不重排布局），没有运行中的节点就自动停。（见 `GraphNode.started_at` / `finished_at` / `duration`、`_format_duration`、`FlowPanel._tick` / `_ensure_ticking`、`FlowNode.apply` 中的 `.blink` 类、`FlowCanvas.repaint_running`）
- **节点消息可展开 / 折叠**：每个节点都**完整保留**自己的消息（工具参数 / 返回、模型结果、通知正文）：默认**折叠**（只显一行预览 + 一个‘▸’箭头），**点击节点**即展开看全文（换成‘▾’）；而 `notice` 节点**默认自动展开**，无需点击就能看到系统信息。展开 / 折叠会改变节点高度，并**自动重排**把下方节点顶开，不会相互覆盖。注意：这种**inline 展开 / 折叠只适用于不可弹窗的节点**（如 tool / notice）；可弹窗的节点（如模型调用）是**弹窗优先**的，既无箭头也不在原地展开（见下条）。（`FlowPanel.toggle` / `set_expanded` / `expand_all` / `collapse_all`，以及 `GraphNode.expandable`）
- **模型调用详情弹窗（输入 / 输出全文）**：每个 **LLM 节点**都会完整捕获这次调用的**输入 utterance（完整 prompt）**与**输出 response（原始回复）**。因为它们往往很长，节点里只做**折叠预览**；节点头部只多一个小按钮 `⤢`（**不再有 inline 展开箭头**）；这类可弹窗的节点是**弹窗优先（popup-only）**的：**点击节点任意位置（头部或正文）都会弹出**一个**覆盖约 90% 屏幕的弹窗**友好展示全文（`INPUT` / `RESPONSE` 分段，可滚动、可选中），而**不再在节点内展开 / 折叠**（两种展开手势分布在两个位置容易混淆）。弹窗会**破例把焦点转移过去**（这是“焦点常驻输入框”的唯一例外），底部有两个**按钮且标签内直接显示快捷键**：`Copy (c)` **一键复制全文到剪切板**、`Close (Esc)` 关闭；`c` / `Esc` / `q` 键也同样可用（`Close (Esc)` 固定在**弹窗右下角**，与其他弹窗保持一致），关闭后焦点自动收回输入框。（`ChatApp.on_flow_panel_inspect_requested` / `InspectModal` / `FlowPanel.open_inspector`）
- **单条消息可点开（双击气泡 → 阅读 / 复制 / 编辑弹窗）**：终端原生拖选复制在部分终端里不稳定，因此每个聊天气泡都可以**点开成弹窗**来可靠地阅读 / 复制。为防误点开，采用**两步式**：**第一次单击**只**选中**这条消息（高亮描边），**第二次点击**已选中的气泡才会弹窗。弹窗**覆盖约 90% 屏幕**，正文放在**只读但可选中**的 `TextArea` 里（即使原生选择失效也能选中文字）；弹窗会**破例把焦点转移过去**（“焦点常驻输入框”的例外）。底部按钮**标签内直接显示快捷键**：`Copy (c)` 复制全文、`Edit & resend (e)`（仅用户消息）先把会话**回退到该消息之前**、再把原文**预填回输入框**，改完重发即**在该处分叉**（等价于 Claude/ChatGPT 的“编辑消息”）、`Close (Esc)` 关闭（固定在**弹窗右下角**，与其他弹窗一致）；`c` / `e` / `Esc` / `q` 也可用。**气泡弹窗本身不再内嵌会话树 / Restore 控件**（旧的内嵌控件因作用不明确而去掉）；取而代之，**恢复到任意节点现在是一个独立功能，从**底栏**打开（见下方“恢复到节点”）；底层类 git 的会话运行时同时**支撑 Edit 与 Restore（回退 + 分叉）**。若在某个回合仍在运行时点击编辑，会先用之前的**任务终止体系**（`CancelToken`）中断，并**await 等待旧回合真正退出**，再回退 + 预填，避免旧回合的响应落进已回退的对话里（出现“没有对应请求的响应”）。**选中一条气泡不会抢走键盘输入**：选中只是视觉高亮，焦点仍钉在输入框。（底层会话树仍在 `coreybot/runtime/session.py`：`SessionTree` / `SessionNode` / `Snapshot`，**追加式** + **每个 agent 一条分支 head**（multi-agent 钩子），供 Edit 的 `checkout_session` 与分叉提交使用，也为**后续文件恢复**预留 `Snapshot.artifacts`；`Agent` 每个回合提交一个节点。`MessageBubble.OpenRequested` → `SessionModal`；编辑走 `_edit_message` → `Agent.checkout_session`）
- **从底栏恢复到任意会话节点（类 git 的节点地图，`Ctrl+R`）**：这是一个**独立功能**，与气泡里的 `Edit` 区分开。点底栏的 **`sessions` 提示**（动作名仍为 `restore`）（或按 `Ctrl+R`）会弹出一个 **`RestoreModal`**，里面把**整棵会话树用和右侧 telemetry 仪表盘一样的风格画出来**：每个提交是一个**圆角边框盒子**，由 `SessionTree.graph_layout()` 定位——**线性历史会笔直向下堆叠在同一列（不再像以前的缩进树那样多轮对话一直往右缩进）**，**分叉才会步进到新的一列**；盒子之间由一个与仪表盘相同的连接层画出正交连线（`│─└▸`）。（连线和右侧仪表盘一致：竖线从**父盒底部边框**出发，拐角落在**子盒左侧中部**，两端都紧贴到盒子上）当前所在节点用**绿色边框 + 实心圆点 `●`** 标出，其余节点用**空心圈 `○`**，分叉点带 `fork` 标记，**被废弃的分支也会一并画出来**。弹窗**破例拿走焦点**（与其他弹窗相同的例外）：**方向键**移动选中光标（按行），**单击**选中某个盒子，**拖动背景**可以平移画布；默认高亮当前节点。**激活**某个节点（回车、再次点击已选中的盒子，或点 `Restore` 按钮）后，**会把整个上下文（history + telemetry）恢复到该节点**。为避免冲突，恢复前会**先用任务终止体系（`CancelToken`）中断，并 await 阻塞等待全部 task 真正终止**后才 checkout，否则旧回合的响应会落进已恢复的对话里（“没有对应请求的响应”）。恢复是**非破坏性**的（git checkout）：你离开的分支仍在，之后再发送就从该节点**分叉**。（另：一次发送只有**完成**后才会在会话树里生成节点，所以**正在进行的回合**会以一个**虚线“待处理”方块**挂在当前节点下方展示；它不是真正的提交，不可选中、不可恢复。）弹窗的 `Close (Esc)` 按钮同样固定在**右下角**，与其他弹窗一致。**这个弹窗分为两个页签（页签做成和按钮一样的样式，与 `sessions` 标题在同一行，紧凑不占多余空间，当前页签高亮为主色按钮）**：第一个页签「**Session tree**」就是上面描述的**当前会话树**（行为不变）；第二个页签「**All sessions**」是**全局会话管理**，管理 home 目录下保存的**所有会话**：左边是**全局会话列表**（按时间倒序，当前会话用绿色 `●` 标出，顶部有一个简单**搜索框**，按标题 / id / 时间子串过滤），右边是**会话记录预览**（按会话时间**平铺展开**，不展示树状结构）；在全局页签里回车 / 双击某个会话就会**把它加载为当前会话**（同样先中断并 await 等待在途回合终止，再切换）；在全局页签上 `s`/`w` 不触发恢复，只是向搜索框输入（见 `ChatApp.action_restore` / `RestoreModal` / `_SessionCanvas` / `SessionTree.graph_layout` / `on_restore_modal_restore_requested` / `Agent.checkout_session`）
- **为未来 DAG 预留的稳定画布**：节点位置**完全由模型推导**——每次变更后由 `FlowPanel._relayout()` 重新计算（纯函数、幂等、按真实高度堆叠不重叠）。目前默认是**纵向单列**布局，但位置计算已与渲染 / 命中测试 / 拖拽完全解耦：将来做**多 Agent / 并行工具调用**需要横向铺开的**有向无环图**时，只需在 `_relayout()` 里换一个按图深度分列的算法（已预留列间距 `_H_GAP`），无需改动画节点 / 连线 / 交互的任何代码；节点高度也是动态的（`height: auto`），所以展开或宽窄不一的节点都不会错位。

- **无顶栏，底部单行状态栏 + 心跳动画**：删除了顶部 Header，把顶部留给**溢出的聊天向上滚动**（不再被一条栏遮住）。底部只用**一行** 同时承载原 Header 的信息（**当前运行目录路径**、`provider · model`）**与**原 Footer 的快捷键提示（`Esc` 中断 · `Ctrl+L` 清屏 · `Ctrl+C` 退出）——不再有单独的 Footer 行。底栏右侧的快捷键提示（`Esc` 中断、`Ctrl+L` 清屏、`Ctrl+C` 退出）**仍可直接点击**（每个提示是一个 `_KeyHint` 小控件，点击会运行与按键相同的动作，悬停时高亮）。**不显示时钟**，而是用一颗**待机呼吸灯**（单个圆点 `●`）表明事件循环‘还活着’：它的亮度会像早期手机息屏后的通知灯那样**平滑地一张一弛（真正的渐变，而不是硬性的开/关闪烁）**。状态同时用**颜色 + 闪烁频率**表达：空闲时是**平静的绿色、呼吸较慢**（`ready`）；回合执行中是**琉珀色（amber）、呼吸更快**（`working`）。呼吸曲线是一个以 `_beat` 为自变量的三角波（在接近关闭的底色与状态色之间线性插值成 `#rrggbb` 真彩色）；新状态（比如错误红色快闪）只需在 `_LED_STATES` 里加一行 `颜色 + 周期` 即可扩展。（见 `ChatApp._led_indicator` / `_status_state` / `_tick_status`）键盘**焦点始终钉在消息输入框 `#prompt` 上**（全局只有这一处需要打字）：点击右侧流程图/节点、或按 `Tab` 把焦点抢走时，`ChatApp.on_descendant_focus` 会立即把焦点弹回输入框（而‘点节点展开/折叠’、‘拖动画布’这些鼠标操作不依赖焦点，照常可用）。（见 `ChatApp.on_descendant_focus` / `_focus_prompt`）焦点常驻输入框**不会影响你选中 / 复制聊天文本**：直接用鼠标**拖选**左侧气泡，再按 `Ctrl+C` 即可复制（复制的是消息**正文**，不包含 `you │` 前缀）。（为此左侧聊天区 `#chat` 设为**不可获得焦点**：否则一按下就会把焦点抢到滚动容器上、触发焦点回弹而**把拖选中途打断**——这正是之前“一选焦点就跳回”的原因；改为不可聚焦后仍可用滾轮滚动）。因为气泡是两列 `Table.grid` 渲染，Textual 默认抽不出可选文本，所以`MessageBubble.get_selection` 做了重写。由于 `Ctrl+C` 的键位链是 `Input 复制 → 屏幕选区复制 → 退出`，所以**有选区时 `Ctrl+C` 复制、无选区时 `Ctrl+C` 仍然退出**。（见 `MessageBubble.get_selection`）

### 事件与来源（可扩展位）

Agent 每一步都会发出带 `source` 字段的 `AgentEvent`（见 `coreybot/runtime/agent.py` 中的 `Source`）。流程图通过 `SOURCE_STYLE`（见 `coreybot/frontends/tui/flow.py`）**按 `source` 查表**决定图标与颜色：

| source   | 图标 | 含义                         |
| -------- | ---- | ---------------------------- |
| `llm`    | 🧠   | 一次模型往返                 |
| `tool`   | 🔧   | 一次工具调用                 |
| `mcp`    | 🔌   | 预留：MCP 服务器             |
| `skill`  | ✨   | 预留：skill                  |
| `agent`  | 🤖   | 预留：声明式 / 子 agent      |
| `system` | •    | 生命周期 / 提示              |

关键在于：`source` 只是一个**字符串**。将来接入 MCP / skill / 声明式 agent 时，只要在发出的事件上打一个新的 `source` 标签，流程图就会**自动**为它渲染对应图标（未知来源也有兜底样式），**无需改动 UI 代码**——这正是这套设计留出的扩展点。

## 异步架构与会话中断

整个框架是 **异步（asyncio）** 的，这样才能在一次请求进行到一半时干净地把它 **中断**——就像在 Go 里把一个 `context.Context` 取消掉。

### 核心：`CancelToken`

`coreybot/core/cancel.py` 提供一个共享的取消信号 `CancelToken`（类比 Go 的 `context`）：

- `is_cancelled` / `raise_if_cancelled()`：线程安全的标志，**阻塞代码**（在线程里跑的 `urllib`）可在分块之间轮询它。
- `await token.wait()`：**异步代码**可 await 它，用 `run_cancellable(coro, token)` 让「真实工作」和「取消信号」赛跑，谁先到就用谁的结果。
- `token.cancel()`：可从任意线程调用，同时唤醒轮询者和 await 者。

为什么需要它？标准库没有异步 HTTP，本项目又坚持少依赖，所以真正的网络 I/O 仍是阻塞的 `urllib`，只是用 `asyncio.to_thread` 挪出事件循环。线程无法被强杀，于是采用 Go 一样的 **协作式取消**：阻塞代码在每个 SSE 分块之间检查 `is_cancelled` 并尽快停下。

### 数据流

```text
前端（TUI/CLI）
  │  持有本回合的 CancelToken
  ▼
Agent.arun_turn(user_input, cancel_token=…)   # 异步 agent 循环
  │  每步都 await，并用 run_cancellable 包住 LLM/工具调用
  ▼
Provider.acomplete / astream(messages, cancel_token=…)   # 异步、可取消
  │
  ▼
http_client.apost_json / apost_sse(…, cancel_token=…)    # 阻塞 urllib 跑在线程里，分块间轮询取消
```

一次取消会让在途的 await 抛出 `CancelledError`，前端把它渲染成「已中断」的提示，而不是崩溃；被放弃的这轮用户消息会从历史里回滚，保持上下文一致。

### 如何中断

- **TUI**：请求进行中按 `Esc`（右侧流程图会显示中断提示，输入框重新可用）。
- **CLI**：请求进行中按 `Ctrl+C`（打印 `(interrupted)`，循环继续，可再次输入）。

> 兼容性：同步的 `provider.complete()` / `agent.run_turn()` 仍保留为便捷封装（内部 `asyncio.run(...)`），供脚本或不在事件循环中的调用使用。

## 目录结构

项目按「职责」分层为若干子包，便于向大型项目成长——每一层只依赖比它更底层的层：

```text
coreybot/
  __main__.py              # 入口（python -m coreybot）：解析参数 / 开会话（--home / --yes）并分发
  core/                    # 跨层复用的基础原语（不依赖其他子包）
    config.py              #   配置 + .env 解析
    message.py             #   统一的 Message / Role / CompletionResult
    cancel.py              #   CancelToken：类 Go context 的协作式取消
    paths.py               #   以户目录下 .coreybot 为根的目录布局（仿 Codex 的 ~/.codex）
  llm/                     # 与大模型对话的一切：协议 + 传输 + provider 适配
    protocol.py            #   与模型的 XML 标签交互协议（解析/构造）
    http_client.py         #   基于 urllib 的极简 JSON + SSE 客户端（含异步、可取消封装）
    providers/
      base.py              #   LLMProvider 抽象基类 + 注册表
      openai_provider.py   #   OpenAI Chat Completions（含流式）
      anthropic_provider.py#   Anthropic Messages API
      gemini_provider.py   #   Gemini generateContent
  runtime/                 # agent 运行时
    agent.py               #   agent 循环（聊天 + 工具调用）
    session.py             #   类 git 的会话树（分叉 / checkout / 快照）
    session_store.py       #   会话树 ↔ JSONL rollout 的无损序列化
    session_service.py     #   粘合层：解析 home + 首次启动确认 + 提供 saver
  frontends/               # 面向用户的前端
    chat_loop.py           #   纯命令行聊天循环
    tui/
      app.py               #   Textual TUI（分屏：左聊天 / 右流程图）
      flow.py              #   FlowPanel：可拖动/自动跟随的节点流程图（按 source 渲染）
  tools/                   # 工具系统（与前端/运行时解耦）
    base.py                #   Tool / ToolRegistry / @tool 装饰器
    builtin/               #   内置工具：自动发现带 tool.py 的子包
      __init__.py          #     扫描并导入各工具子包（无需中心登记）
      calc/                #     spec.py（接口）+ tool.py（实现）+ tests/
      clock/               #     spec.py + tool.py + tests/
      read_file/           #     spec.py + tool.py + tests/
tests/                     # pytest 单元测试 + 集成测试（面向包的公共 API）
```

> 依赖方向：core -> llm -> runtime -> frontends；tools 独立于前端与运行时，
> 由 runtime 通过注册表使用。层内一律用相对导入，跨层用绝对导入
> （如 from coreybot.core.config import Config），这样移动单个文件不会牵连全局。

## 环境要求

- Python 3.11+（本项目基于 3.14 开发）。
- 运行依赖：`textual`（仅 TUI 需要）。
- 测试依赖：`pytest`、`pytest-asyncio`。

## 安装

创建虚拟环境并安装依赖。Windows PowerShell：

```powershell
# 在项目根目录执行
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install textual pytest pytest-asyncio
```

macOS / Linux：

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install textual pytest pytest-asyncio
```

## 配置

配置从环境变量读取（也可放到项目根目录的 `.env` 文件）。复制示例文件并按需修改：

```powershell
Copy-Item coreybot\.env.example .env
```

| 变量                | 默认值                                         | 说明                                |
| ------------------- | ---------------------------------------------- | ----------------------------------- |
| `LLM_PROVIDER`      | `openai`                                       | `openai` \| `anthropic` \| `gemini` |
| `LLM_BASE_URL`      | `http://127.0.0.1:23333/api/openai/v1`         | API 基础地址                        |
| `LLM_API_KEY`       | `not-needed`                                   | API 密钥 / 令牌                     |
| `LLM_MODEL`         | `claude-opus-4.8`                              | 模型名称                            |
| `LLM_SYSTEM_PROMPT` | `You are a helpful assistant.`                 | 系统提示词                          |
| `LLM_TIMEOUT`       | `60`                                           | 请求超时（秒）                      |

真实的环境变量始终优先于 `.env` 文件中的值。

## 运行

启动 **TUI**（默认，分屏：左聊天 / 右实时流程图）：

```powershell
.\.venv\Scripts\python.exe -m coreybot
```

改用纯 **CLI** 循环：

```powershell
.\.venv\Scripts\python.exe -m coreybot --cli
```

通过命令行覆盖 provider / 模型 / 基础地址：

```powershell
.\.venv\Scripts\python.exe -m coreybot --provider anthropic --model claude-opus-4.8
.\.venv\Scripts\python.exe -m coreybot --base-url http://127.0.0.1:23333/api/openai/v1
# 指定会话存储目录（优先于 $COREYBOT_HOME），首次启动时 --yes 免确认直接创建：
.\.venv\Scripts\python.exe -m coreybot --home D:\agent-home --yes
```

## 会话持久化（以用户目录的 .coreybot 为根）

agent 把所有持久化状态集中存在一个 **home 目录**里，布局仿照 Codex 的 `~/.codex`。默认位置是 `~/.coreybot`，由静态目录名 `.coreybot` 在**运行时**拼接到当前用户目录得到（不在导入时冻结，总是跟随当前用户）。

**覆盖优先级**（高→低）：

1. 代码 / 命令行显式指定：`--home <path>`（或 `open_session(override=...)`）；
2. 环境变量 `COREYBOT_HOME`（非空时生效，类似 Codex 的 `CODEX_HOME`）；
3. 默认 `~/.coreybot`。

**目录布局**（也是 Codex 形状）：

```text
<home>/                                          # 默认 ~/.coreybot
  config.toml                                    # 用户可改的设置
  version.json                                   # 写入者 + 时间戳
  history.jsonl                                  # 跨会话的输入历史
  sessions/YYYY/MM/DD/rollout-<ts>-<id>.jsonl    # 每个会话一个文件（按日期分桶）
  logs/
```

**首次启动确认**：当 home 目录不存在时，程序会先征求同意再创建（终端上一行 `y/N` 提示，默认 Yes）：

- 加 `--yes` / `-y`：不询问直接创建；
- 非交互环境（管道 / 测试）：自动创建，不会阻塞；
- 选择拒绝：程序照常运行，但**不写盘**（不会创建任何目录）。

**会话树落盘**：每次回合提交、`clear` 新根、以及恢复（checkout）后，agent 都会把整棵会话树写到本次运行的 rollout 文件（JSONL）。落盘是**尽力而为**的：I/O 出错不会中断一次回合。写盘格式与 Codex rollout 一致：首行 `meta`，其余每行一个 `node`；加载时**不重放** commit（那会重新分配 id）而是精确重建 id / heads / roots / 计数器，以保证“恢复到节点”稳定。

> 设计上 `session.py`（纯内存会话树）与 `session_store.py`（序列化）、`session_service.py`（粘合 + 确认）分层解耦：会话树本身零持久化耦合。`Snapshot.artifacts` 预留了文件快照位（已编辑 / 已删除文件的恢复），日后可在不动树的前提下叠加。

### 应用内命令与快捷键

- `/help` — 显示帮助
- `/reset`（同 `Ctrl+L`）— 清空当前对话上下文（保留系统提示词）；**不会丢掉会话树**——不再新建一个孤立的根，而是**从最初的根节点分叉出一条新线**（`SessionTree.new_root`，新节点的 parent 就是根），所以根节点会**分叉成两个子节点**：一个是清空前的对话，另一个是清空后的新对话；旧分支仍在会话树里，仍可在 `sessions`（会话管理）里**恢复到清空前的任意节点**（注：**会话面板的流程图不会显示 `clear` 节点本身**，因为 clear 不会恢复任何上下文，不是一个有意义的恢复目标；清空后的新对话会直接挂在最初的根节点下作为一个分叉展示；底层会话树与持久化仍保留该节点）
- `/history` — 打印原始消息历史
- `/tools` — 列出可用工具
- `/exit` — 退出
- 快捷键（TUI）：`Esc` 中断当前请求，`Ctrl+R` 打开 `sessions` 会话管理（两个页签：当前会话节点地图 + 全局会话浏览/搜索/预览），`Ctrl+L` 清空上下文（保留会话树，可恢复），`Ctrl+C` 退出
- CLI：请求进行中按 `Ctrl+C` 中断当前请求（循环不退出）

## 测试

运行全部测试：

```powershell
.\.venv\Scripts\python.exe -m pytest
```

只跑单元测试（更快）或只跑集成测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -m "not integration"
.\.venv\Scripts\python.exe -m pytest -m integration
```

以详细模式运行单个文件：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_providers_gemini.py -v
```

只跑某个内置工具（测试与实现同目录）：

```powershell
.\.venv\Scripts\python.exe -m pytest coreybot\tools\builtin\calc
```

> 测试目录：`pytest.ini` 的 `testpaths = tests coreybot`，既收集 `tests/` 下的测试，也收集各工具子包 `coreybot/tools/builtin/<名字>/tests/` 里的**就近测试**。根目录的 `conftest.py` 会把共享 fixture（如 `local_tmp_path`）暴露给整棵目录树。

测试覆盖内容：

- `tests/test_message.py`、`test_config.py`、`test_registry.py` — 核心数据模型、配置 /`.env` 解析、provider 注册表。
- `tests/test_http_client.py` — 启动一个真实的本地 HTTP 服务器，测试同步与**异步/可取消**的 `post_json` / `post_sse`。
- `tests/test_providers_*.py` — 各协议的请求结构与响应解析（同时充当协议文档）。
- `tests/test_tools.py` — 工具**系统**本身（注册表、`@tool` 装饰器、参数校验、目录渲染）。
- `coreybot/tools/builtin/*/tests/` — 每个内置工具的**就近单元测试**（calc 的安全性、read_file 的边界等）。
- `coreybot/tools/builtin/tests/test_builtin_conventions.py` — 内置工具的**目录规范强制校验**（相当于项目自带的 lint）：逐条断言 `spec.py` / `tool.py` / `tests/` 布局与 `@tool(spec=SPEC)` 写法，规范详见 `coreybot/tools/builtin/README.md`。
- `tests/test_cancel.py` — `CancelToken` / `run_cancellable` 与 agent 级中断。
- `tests/test_tui.py` — Textual 无头 `Pilot` 测试（命令、流式、`Esc` 中断、错误处理）。
- `tests/test_chat_loop.py` — 打桩 `input`/`print` 的命令行循环测试。

## 本机环境说明

本机的 `%TEMP%` 目录被安全软件锁定，会导致 pytest 默认的临时目录 / 缓存目录报错。项目已内置规避方案：

- `pytest.ini` 设置 `addopts = -p no:cacheprovider`，关闭磁盘缓存。
- 测试使用项目内的 `local_tmp_path` fixture（位于 `tests/.artifacts`）替代内置的 `tmp_path`。

因此直接运行 `pytest` 即可，无需设置任何额外环境变量。