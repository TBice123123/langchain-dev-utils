# 为工具调用添加人工审核

## 概述

本库提供了装饰器函数，用于为工具调用添加"人在回路"审核支持，在工具执行期间启用人工审核。

| 装饰器 | 适用场景 |
|--------|----------|
| `human_in_the_loop` | 用于同步工具函数 |
| `human_in_the_loop_async` | 用于异步工具函数 |

## 使用示例

### 使用默认的 handler

```python
from langchain_dev_utils.tool_calling import human_in_the_loop
import datetime


@human_in_the_loop
def get_current_time() -> str:
    """获取当前时间戳"""
    return str(datetime.datetime.now().timestamp())
```

### 异步工具示例

```python
from langchain_dev_utils.tool_calling import human_in_the_loop_async
import asyncio
import datetime


@human_in_the_loop_async
async def async_get_current_time() -> str:
    """异步获取当前时间戳"""
    await asyncio.sleep(1)
    return str(datetime.datetime.now().timestamp())
```

### 默认 handler 的实现

默认 handler 的实现如下：

```python
def _get_human_in_the_loop_request(params: InterruptParams) -> dict[str, Any]:
    return {
        "action_request": {
            "action": params["tool_call_name"],
            "args": params["tool_call_args"],
        },
        "config": {
            "allow_accept": True,
            "allow_edit": True,
            "allow_respond": True,
        },
        "description": f"Please review tool call: {params['tool_call_name']}",
    }


def default_handler(params: InterruptParams) -> Any:
    request = _get_human_in_the_loop_request(params)
    response = interrupt(request)

    if response["type"] == "accept":
        return params["tool"].invoke(params["tool_call_args"])
    elif response["type"] == "edit":
        updated_args = response["args"]
        return params["tool"].invoke(updated_args)
    elif response["type"] == "response":
        return response["args"]
    else:
        raise ValueError(f"Unsupported interrupt response type: {response['type']}")
```

#### 中断请求格式

中断时会发送如下 JSON Schema 格式的请求：

| 字段 | 说明 |
|------|------|
| `action_request.action` | 工具调用名称。<br><br>**类型**: `str` |
| `action_request.args` | 工具调用参数。<br><br>**类型**: `dict` |
| `config.allow_accept` | 是否允许接受操作。<br><br>**类型**: `bool` |
| `config.allow_edit` | 是否允许编辑参数。<br><br>**类型**: `bool` |
| `config.allow_respond` | 是否允许直接响应。<br><br>**类型**: `bool` |
| `description` | 操作描述。<br><br>**类型**: `str` |

#### 中断响应格式

响应时需要返回如下 JSON Schema 格式的数据：

| 字段 | 说明 |
|------|------|
| `type` | 响应类型，可选值为 `accept`、`edit`、`response`。<br><br>**类型**: `str`<br>**必填**: 是 |
| `args` | 当 `type` 为 `edit` 或 `response` 时，包含更新后的参数或响应内容。<br><br>**类型**: `dict`<br>**必填**: 否 |



### 自定义 Handler 示例

你可以完全控制中断行为，例如只允许"接受/拒绝"，或自定义提示语：

```python
from typing import Any
from langchain_dev_utils.tool_calling import human_in_the_loop_async, InterruptParams
from langgraph.types import interrupt


async def custom_handler(params: InterruptParams) -> Any:
    response = interrupt(
        f"我要调用工具 {params['tool_call_name']}，参数为 {params['tool_call_args']}，请确认是否调用"
    )
    if response["type"] == "accept":
        return await params["tool"].ainvoke(params["tool_call_args"])
    elif response["type"] == "reject":
        return "用户拒绝调用该工具"
    else:
        raise ValueError(f"不支持的响应类型: {response['type']}")


@human_in_the_loop_async(handler=custom_handler)
async def get_weather(city: str) -> str:
    """获取天气信息"""
    return f"{city}天气晴朗"
```

!!! success "最佳实践"
    该装饰器在实现自定义人在回路逻辑时，需要传入 `handler` 参数。此 `handler` 参数是一个函数，内部必须使用 LangGraph 的 `interrupt` 函数来执行中断操作。因此，如果仅为单个工具添加自定义的人在回路逻辑，建议直接使用 LangGraph 的 `interrupt` 函数。当多个工具需要相同自定义人在回路逻辑时，使用本装饰器可以有效避免代码重复。
