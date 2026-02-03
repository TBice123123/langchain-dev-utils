# Adding Human-in-the-Loop Review for Tool Calling

## Overview

This library provides decorator functions for adding human-in-the-loop review support to tool calls, enabling human review during tool execution.

| Decorator | Applicable Scenario |
|-----------|----------------------|
| `human_in_the_loop` | For synchronous tool functions |
| `human_in_the_loop_async` | For asynchronous tool functions |

## Usage Examples

### Using the Default Handler

```python
from langchain_dev_utils.tool_calling import human_in_the_loop
import datetime


@human_in_the_loop
def get_current_time() -> str:
    """Get the current timestamp"""
    return str(datetime.datetime.now().timestamp())
```

### Asynchronous Tool Example

```python
from langchain_dev_utils.tool_calling import human_in_the_loop_async
import asyncio
import datetime


@human_in_the_loop_async
async def async_get_current_time() -> str:
    """Asynchronously get the current timestamp"""
    await asyncio.sleep(1)
    return str(datetime.datetime.now().timestamp())
```

### Implementation of the Default Handler

The default handler is implemented as follows:

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

#### Interrupt Request Format

During an interrupt, a request in the following JSON Schema format is sent:

| Field | Description |
|-------|-------------|
| `action_request.action` | Tool call name.<br><br>**Type**: `str` |
| `action_request.args` | Tool call arguments.<br><br>**Type**: `dict` |
| `config.allow_accept` | Whether to allow the accept action.<br><br>**Type**: `bool` |
| `config.allow_edit` | Whether to allow editing arguments.<br><br>**Type**: `bool` |
| `config.allow_respond` | Whether to allow direct response.<br><br>**Type**: `bool` |
| `description` | Action description.<br><br>**Type**: `str` |

#### Interrupt Response Format

The response must return data in the following JSON Schema format:

| Field | Description |
|-------|-------------|
| `type` | Response type, with possible values `accept`, `edit`, `response`.<br><br>**Type**: `str`<br>**Required**: Yes |
| `args` | When `type` is `edit` or `response`, contains the updated arguments or response content.<br><br>**Type**: `dict`<br>**Required**: No |



### Custom Handler Example

You can fully control the interrupt behavior, such as only allowing "accept/reject", or customizing the prompt:

```python
from typing import Any
from langchain_dev_utils.tool_calling import human_in_the_loop_async, InterruptParams
from langgraph.types import interrupt


async def custom_handler(params: InterruptParams) -> Any:
    response = interrupt(
        f"I am about to call tool {params['tool_call_name']} with arguments {params['tool_call_args']}. Please confirm whether to proceed."
    )
    if response["type"] == "accept":
        return await params["tool"].ainvoke(params["tool_call_args"])
    elif response["type"] == "reject":
        return "User rejected calling this tool"
    else:
        raise ValueError(f"Unsupported response type: {response['type']}")


@human_in_the_loop_async(handler=custom_handler)
async def get_weather(city: str) -> str:
    """Get weather information"""
    return f"The weather in {city} is sunny."
```

!!! success "Best Practice"
    To implement custom human-in-the-loop logic with this decorator, you need to pass a `handler` parameter. This `handler` parameter is a function that must internally use LangGraph's `interrupt` function to perform the interrupt operation. Therefore, if you only need to add custom human-in-the-loop logic for a single tool, it is recommended to use LangGraph's `interrupt` function directly. When multiple tools require the same custom human-in-the-loop logic, using this decorator can effectively avoid code duplication.