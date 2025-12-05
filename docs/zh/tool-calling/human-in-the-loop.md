# 添加人在回路支持

> [!NOTE]
>
> **功能概述**：提供方便为工具调用过程添加“人在回路”审核支持的实用工具。
>
> **前置要求**：了解 langchain 的[Tool](https://docs.langchain.com/oss/python/langchain/tools)、[人在环路](https://docs.langchain.com/oss/python/langchain/human-in-the-loop)。
>
> **预计阅读时间**：5 分钟

## 概述

提供装饰器函数，用于为工具调用添加“人在回路”审核支持，在工具执行期间启用人工审核。
具体表现为两个装饰器：
- `human_in_the_loop`：用于同步工具函数
- `human_in_the_loop_async`：用于异步工具函数

其参数如下:

<Params
name="func"
type="Callable"
description="待装饰的函数（请勿手动传参，仅用于装饰器语法）"
:required="true"
:default="null"
/>
<Params
name="handler"
type="Callable[[InterruptParams], Any]"
description="可选，自定义中断处理逻辑。若未提供，则使用内置 default_handler。对于异步装饰器，handler 必须是异步函数。"
:required="false"
:default="null"
/>

## 使用示例

### 使用默认的 handler

```python
from langchain_dev_utils import human_in_the_loop
from langchain_core.tools import tool
import datetime

@human_in_the_loop
@tool
def get_current_time() -> str:
    """获取当前时间戳"""
    return str(datetime.datetime.now().timestamp())
```

### 异步工具示例

```python
from langchain_dev_utils import human_in_the_loop_async
from langchain_core.tools import tool
import asyncio
import datetime

@human_in_the_loop_async
@tool
async def async_get_current_time() -> str:
    """异步获取当前时间戳"""
    await asyncio.sleep(1)
    return str(datetime.datetime.now().timestamp())
```

::: details 默认的 handler

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

:::

### 自定义 Handler 示例

你可以完全控制中断行为，例如只允许“接受/拒绝”，或自定义提示语：

```python
from typing import Any
from langchain_dev_utils import human_in_the_loop_async, InterruptParams
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
@tool
async def get_weather(city: str) -> str:
    """获取天气信息"""
    return f"{city}天气晴朗"
```

<BestPractice>
该装饰器在实现自定义人在回路逻辑时，需要传入handler参数。此handler参数是一个函数，内部必须使用LangGraph的interrupt函数来执行中断操作。因此，如果仅为单个工具添加自定义的人在回路逻辑，建议直接使用LangGraph的interrupt函数。当多个工具需要相同自定义人在回路逻辑时，使用本装饰器可以有效避免代码重复。
</BestPractice>
