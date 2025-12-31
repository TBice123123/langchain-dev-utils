# Tool Call Processing

## Overview

Provides utilities for detecting and parsing tool call arguments.

## Detect Tool Calls

Detects whether a message contains a tool call. The core function is `has_tool_calling`.

### Code Example

```python
import datetime
from langchain_core.tools import tool
from langchain_dev_utils.tool_calling import has_tool_calling

@tool
def get_current_time() -> str:
    """Get the current timestamp"""
    return str(datetime.datetime.now().timestamp())

response = model.bind_tools([get_current_time]).invoke("What time is it now?")
print(has_tool_calling(response))
```

## Parse Tool Call Arguments

Provides a utility function to parse tool call arguments, extracting parameter information from a message. The core function is `parse_tool_calling`.

### Code Example

```python
import datetime
from langchain_core.tools import tool
from langchain_dev_utils.tool_calling import has_tool_calling, parse_tool_calling

@tool
def get_current_time() -> str:
    """Get the current timestamp"""
    return str(datetime.datetime.now().timestamp())

response = model.bind_tools([get_current_time]).invoke("What time is it now?")

if has_tool_calling(response):
    name, args = parse_tool_calling(
        response, first_tool_call_only=True
    )
    print(name, args)
```