# 为工具调用添加人工审核

## 概述

提供装饰器函数，用于为工具调用添加“人在回路”审核支持，在工具执行期间启用人工审核。
具体表现为两个装饰器：

- `human_in_the_loop`：用于同步工具函数

- `human_in_the_loop_async`：用于异步工具函数

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

!!! note "默认的 handler"
    如下是默认的 handler 实现：
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

    中断的时候会发送一个 JSON Schema 内容如上`_get_human_in_the_loop_request`返回的值,回复的时候需要返回一个 JSON Schema 内容，要有一个键为 `type`，值为 `accept`/`edit`/`response`。


### 自定义 Handler 示例

你可以完全控制中断行为，例如只允许“接受/拒绝”，或自定义提示语：

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

!!! note "最佳实践"
    该装饰器在实现自定义人在回路逻辑时，需要传入handler参数。此handler参数是一个函数，内部必须使用LangGraph的interrupt函数来执行中断操作。因此，如果仅为单个工具添加自定义的人在回路逻辑，建议直接使用LangGraph的interrupt函数。当多个工具需要相同自定义人在回路逻辑时，使用本装饰器可以有效避免代码重复。
