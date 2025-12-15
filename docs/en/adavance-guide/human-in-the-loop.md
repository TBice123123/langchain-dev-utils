# Adding Human Review for Tool Calls

## Overview

This provides decorator functions to add "human-in-the-loop" review support for tool calls, enabling human review during tool execution. This is implemented as two decorators:

- `human_in_the_loop`: For synchronous tool functions

- `human_in_the_loop_async`: For asynchronous tool functions

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

!!! note "Default Handler"
    Below is the implementation of the default handler:
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

    When interrupted, a JSON Schema with the structure returned by `_get_human_in_the_loop_request` is sent. The response must also be a JSON Schema with a key named `type`, and its value should be `accept`/`edit`/`response`.


### Custom Handler Example

You can have full control over the interrupt behavior, for example, by only allowing "accept/reject" or by customizing the prompt:

```python
from typing import Any
from langchain_dev_utils.tool_calling import human_in_the_loop_async, InterruptParams
from langgraph.types import interrupt


async def custom_handler(params: InterruptParams) -> Any:
    response = interrupt(
        f"I am about to call tool {params['tool_call_name']} with arguments {params['tool_call_args']}. Please confirm if I should proceed."
    )
    if response["type"] == "accept":
        return await params["tool"].ainvoke(params["tool_call_args"])
    elif response["type"] == "reject":
        return "The user rejected the tool call."
    else:
        raise ValueError(f"Unsupported response type: {response['type']}")


@human_in_the_loop_async(handler=custom_handler)
async def get_weather(city: str) -> str:
    """Get weather information"""
    return f"The weather in {city} is sunny."
```

!!! success "Best Practice"
    When implementing custom human-in-the-loop logic, you need to pass a `handler` parameter to this decorator. This handler parameter is a function that must internally use LangGraph's `interrupt` function to perform the interruption. Therefore, if you are only adding custom human-in-the-loop logic to a single tool, it is recommended to use LangGraph's `interrupt` function directly. When multiple tools require the same custom human-in-the-loop logic, using this decorator can effectively avoid code duplication.