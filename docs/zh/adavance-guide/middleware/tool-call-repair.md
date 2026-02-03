# 工具调用修复

`ToolCallRepairMiddleware` 是一个用于**自动修复大模型无效工具调用**的中间件。

大模型在输出符合 JSON Schema 的工具调用参数时，有时会因模型能力限制生成格式错误的 JSON 内容（错误通常出现在 `arguments` 字段）。这类解析失败的调用会被 LangChain 标记并存入 `invalid_tool_calls` 字段中。`ToolCallRepairMiddleware` 会自动检测该字段，并调用 `json-repair` 库尝试修复格式，使工具调用能够正常执行。

!!! warning "使用须知"
    使用本中间件前，请确保已安装 `langchain-dev-utils[standard]`，详见**安装指南**。

## 参数说明

该中间件设计为**零配置开箱即用**，实例化时无需传入任何参数。

## 使用示例

### 标准用法

```python
from langchain_dev_utils.agents.middleware import ToolCallRepairMiddleware

agent = create_agent(
    model="vllm:qwen3-4b",
    tools=[run_python_code, get_current_time],
    middleware=[
        ToolCallRepairMiddleware()
    ],
)
```

### 便捷用法（推荐）

由于 `ToolCallRepairMiddleware` 实例化时无需配置参数，本库预置了一个全局实例 `tool_call_repair`，推荐直接使用以简化代码：

```python
from langchain_dev_utils.agents.middleware import tool_call_repair

agent = create_agent(
    model="vllm:qwen3-4b",
    tools=[run_python_code, get_current_time],
    middleware=[tool_call_repair],
)
```

!!! warning "注意事项"
    本中间件无法保证 100% 修复所有无效工具调用，实际修复效果取决于 `json-repair` 库的能力；此外，它仅作用于 `invalid_tool_calls` 字段中的无效调用内容。