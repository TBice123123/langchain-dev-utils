# Tool Calling Module API Reference Documentation

## has_tool_calling

Checks if a message contains a tool call.

### Function Signature

```python
def has_tool_calling(
    message: AIMessage
) -> bool
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| message | AIMessage | Yes | - | The message to check |

### Example

```python
if has_tool_calling(response):
    # Handle tool call
    pass
```

---

## parse_tool_calling

Parses tool call arguments from a message.

### Function Signature

```python
def parse_tool_calling(
    message: AIMessage, first_tool_call_only: bool = False
) -> Union[tuple[str, dict], list[tuple[str, dict]]]
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| message | AIMessage | Yes | - | The message to parse |
| first_tool_call_only | bool | No | False | Whether to return only the first tool call |

### Example

```python
# Get all tool calls
tool_calls = parse_tool_calling(response)

# Get only the first tool call
name, args = parse_tool_calling(response, first_tool_call_only=True)
```

---

## human_in_the_loop

A decorator to add "human-in-the-loop" manual review capability to **synchronous tool functions**.

### Function Signature

```python
def human_in_the_loop(
    func: Optional[Callable] = None,
    *,
    handler: Optional[HumanInterruptHandler] = None
) -> Union[Callable[[Callable], BaseTool], BaseTool]
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| func | Optional[Callable] | No | None | The synchronous function to be decorated (decorator syntactic sugar) |
| handler | Optional[HumanInterruptHandler] | No | None | Custom interrupt handler function |

### Example

```python
@human_in_the_loop
def get_current_time():
    """Get the current time"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

---

## human_in_the_loop_async

A decorator to add "human-in-the-loop" manual review capability to **asynchronous tool functions**.

### Function Signature

```python
def human_in_the_loop_async(
    func: Optional[Callable] = None,
    *,
    handler: Optional[HumanInterruptHandler] = None
) -> Union[Callable[[Callable], BaseTool], BaseTool]
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| func | Optional[Callable] | No | None | The asynchronous function to be decorated (decorator syntactic sugar) |
| handler | Optional[HumanInterruptHandler] | No | None | Custom interrupt handler function |

### Example

```python
@human_in_the_loop_async
async def get_current_time():
    """Get the current time"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

---

## InterruptParams

The type of parameters passed to the interrupt handler function.

### Class Definition

```python
class InterruptParams(TypedDict):
    tool_call_name: str
    tool_call_args: Dict[str, Any]
    tool: BaseTool
```

### Field Description

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| tool_call_name | str | Yes | The name of the tool call |
| tool_call_args | Dict[str, Any] | Yes | The arguments of the tool call |
| tool | BaseTool | Yes | The tool instance |

---

## HumanInterruptHandler

Type alias for the interrupt handler function.

### Type Definition

```python
HumanInterruptHandler = Callable[[InterruptParams], Any]
```