# Tool Call Repair

`ToolCallRepairMiddleware` is a middleware designed to **automatically repair invalid tool calls from large language models**.

When large language models output tool call parameters that conform to a JSON Schema, they may sometimes generate malformed JSON content (often in the `arguments` field) due to model limitations. Such failed parsing attempts are flagged by LangChain and stored in the `invalid_tool_calls` field. `ToolCallRepairMiddleware` automatically detects this field and uses the `json-repair` library to attempt to fix the formatting, allowing the tool calls to execute normally.

!!! warning "Usage Notes"
    Before using this middleware, ensure `langchain-dev-utils[standard]` is installed. Refer to the **Installation Guide** for details.

## Parameter Description

This middleware is designed for **zero-configuration out-of-the-box use**. No parameters are required during instantiation.

## Usage Examples

### Standard Usage

```python
from langchain_dev_utils.agents.middleware import ToolCallRepairMiddleware

agent = create_agent(
    model="vllm:qwen2.5-7b",
    tools=[run_python_code, get_current_time],
    middleware=[
        ToolCallRepairMiddleware()
    ],
)
```

### Convenient Usage (Recommended)

Since instantiating `ToolCallRepairMiddleware` requires no configuration parameters, this library provides a pre-configured global instance `tool_call_repair`. It is recommended to use this directly to simplify your code:

```python
from langchain_dev_utils.agents.middleware import tool_call_repair

agent = create_agent(
    model="vllm:qwen2.5-7b",
    tools=[run_python_code, get_current_time],
    middleware=[tool_call_repair],
)
```

!!! warning "Important Notes"
    This middleware cannot guarantee 100% repair of all invalid tool calls. The actual repair effectiveness depends on the capabilities of the `json-repair` library. Furthermore, it only operates on the invalid calls within the `invalid_tool_calls` field.