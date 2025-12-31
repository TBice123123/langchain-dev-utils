# Add Human Review for Tool Calls

## Overview

This library provides decorator functions to add "human-in-the-loop" review support for tool calls, enabling human review during tool execution.

| Decorator | Applicable Scenario |
|-----------|---------------------|
| `human_in_the_loop` | For synchronous tool functions |
| `human_in_the_loop_async` | For asynchronous tool functions |

## Parameter Description

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `handler` | `Callable` | No | `None` | A custom handler function. If `None`, the default handler will be used. |

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

### Async Tool Example

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

### Default Handler Implementation

The implementation of the default handler is as follows:

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

| Field | Type | Description |
|-------|------|-------------|
| `action_request.action` | `str` | The name of the tool call |
| `action_request.args` | `dict` | The arguments of the tool call |
| `config.allow_accept` | `bool` | Whether to allow the action to be accepted |
| `config.allow_edit` | `bool` | Whether to allow the arguments to be edited |
| `config.allow_respond` | `bool` | Whether to allow a direct response |
| `description` | `str` | A description of the action |

#### Interrupt Response Format

The response should be returned in the following JSON Schema format:

| Field | Type | Description |
|-------|------|-------------|
| `type` | `str` | The response type. Possible values are `accept`, `edit`, `response` |
| `args` | `dict` | When `type` is `edit` or `response`, this contains the updated arguments or response content |

### Custom Handler Example

You can have full control over the interrupt behavior, for example, by only allowing "accept/reject" or by customizing the prompt:

```python
from typing import Any
from langchain_dev_utils.tool_calling import human_in_the_loop_async, InterruptParams
from langgraph.types import interrupt


async def custom_handler(params: InterruptParams) -> Any:
    response = interrupt(
        f"I am about to call the tool {params['tool_call_name']} with arguments {params['tool_call_args']}. Please confirm if I should proceed."
    )
    if response["type"] == "accept":
        return await params["tool"].ainvoke(params["tool_call_args"])
    elif response["type"] == "reject":
        return "The user rejected calling this tool."
    else:
        raise ValueError(f"Unsupported response type: {response['type']}")


@human_in_the_loop_async(handler=custom_handler)
async def get_weather(city: str) -> str:
    """Get weather information"""
    return f"The weather in {city} is sunny."
```

!!! success "Best Practice"
    When implementing custom human-in-the-loop logic with this decorator, you must pass a `handler` parameter. This `handler` is a function that must use LangGraph's `interrupt` function to perform the interruption. Therefore, if you are only adding custom human-in-the-loop logic to a single tool, it is recommended to use LangGraph's `interrupt` function directly. When multiple tools require the same custom human-in-the-loop logic, using this decorator can effectively avoid code duplication.