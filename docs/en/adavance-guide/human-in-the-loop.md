# Adding Human Review for Tool Calls

## Overview

Provides decorator functions to add "human-in-the-loop" review support for tool calls, enabling human review during tool execution.
This is implemented as two decorators:

- `human_in_the_loop`: For synchronous tool functions

- `human_in_the_loop_async`: For asynchronous tool functions

## Usage Examples

### Using the Default Handler

```python
from langchain_dev_utils import human_in_the_loop
from langchain_core.tools import tool
import datetime

@human_in_the_loop
@tool
def get_current_time() -> str:
    """Get the current timestamp"""
    return str(datetime.datetime.now().timestamp())
```

### Async Tool Example

```python
from langchain_dev_utils import human_in_the_loop_async
from langchain_core.tools import tool
import asyncio
import datetime

@human_in_the_loop_async
@tool
async def async_get_current_time() -> str:
    """Asynchronously get the current timestamp"""
    await asyncio.sleep(1)
    return str(datetime.datetime.now().timestamp())
```

!!! note "Default Handler"
    The following is the default handler implementation:
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

    When an interrupt occurs, a JSON Schema content like the value returned by `_get_human_in_the_loop_request` above is sent. When replying, a JSON Schema content needs to be returned, which must have a key `type` with a value of `accept`/`edit`/`response`.

### Custom Handler Example

You can fully control the interrupt behavior, for example, by only allowing "accept/reject" or by customizing the prompt:

```python
from typing import Any
from langchain_dev_utils import human_in_the_loop_async, InterruptParams
from langgraph.types import interrupt

async def custom_handler(params: InterruptParams) -> Any:
    response = interrupt(
        f"I am about to call the tool {params['tool_call_name']} with arguments {params['tool_call_args']}. Please confirm if I should proceed."
    )
    if response["type"] == "accept":
        return await params["tool"].ainvoke(params["tool_call_args"])
    elif response["type"] == "reject":
        return "User rejected the tool call"
    else:
        raise ValueError(f"Unsupported response type: {response['type']}")

@human_in_the_loop_async(handler=custom_handler)
@tool
async def get_weather(city: str) -> str:
    """Get weather information"""
    return f"The weather in {city} is sunny."
```

!!! note "Best Practice"
    When implementing custom human-in-the-loop logic with this decorator, you need to pass the `handler` parameter. This `handler` parameter is a function that must internally use LangGraph's `interrupt` function to perform the interrupt operation. Therefore, if you are only adding custom human-in-the-loop logic to a single tool, it is recommended to directly use LangGraph's `interrupt` function. When multiple tools require the same custom human-in-the-loop logic, using this decorator can effectively avoid code duplication.