# Tool Calling Module API Reference

## has_tool_calling

Checks if a message contains tool calls.

```python
def has_tool_calling(
    message: AIMessage
) -> bool
```

**Parameters:**

- `message`: AIMessage type, required. The message to be checked.

**Return Value:** Boolean type, returns True if the message contains tool calls, otherwise returns False.

**Example:**

```python
if has_tool_calling(response):
    # Handle tool calls
    pass
```

## parse_tool_calling

Parses tool call arguments from a message.

```python
def parse_tool_calling(
    message: AIMessage, first_tool_call_only: bool = False
) -> Union[tuple[str, dict], list[tuple[str, dict]]]
```

**Parameters:**

- `message`: AIMessage type, required. The message to be parsed.
- `first_tool_call_only`: Boolean type, optional. Whether to return only the first tool call, defaults to False.

**Return Value:** A tuple of tool call name and parameters, or a list of tuples containing tool call names and parameters.

**Example:**

```python
# Get all tool calls
tool_calls = parse_tool_calling(response)

# Get only the first tool call
name, args = parse_tool_calling(response, first_tool_call_only=True)
```

## human_in_the_loop

A decorator that adds "human-in-the-loop" manual review capability to **synchronous tool functions**.

```python
def human_in_the_loop(
    func: Optional[Callable] = None,
    *,
    handler: Optional[HumanInterruptHandler] = None
) -> Union[Callable[[Callable], BaseTool], BaseTool]
```

**Parameters:**

- `func`: Optional callable type. The synchronous function to be decorated (decorator syntax sugar).
- `handler`: Optional HumanInterruptHandler type. Custom interrupt handler function.

**Return Value:** BaseTool type, the decorated tool instance.

**Example:**

```python
@human_in_the_loop
def get_current_time():
    """Get current time"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

## human_in_the_loop_async

A decorator that adds "human-in-the-loop" manual review capability to **asynchronous tool functions**.

```python
def human_in_the_loop_async(
    func: Optional[Callable] = None,
    *,
    handler: Optional[HumanInterruptHandler] = None
) -> Union[Callable[[Callable], BaseTool], BaseTool]
```

**Parameters:**

- `func`: Optional callable type. The asynchronous function to be decorated (decorator syntax sugar).
- `handler`: Optional HumanInterruptHandler type. Custom interrupt handler function.

**Return Value:** BaseTool type, the decorated asynchronous tool instance.

**Example:**

```python
@human_in_the_loop_async
async def get_current_time():
    """Get current time"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

## InterruptParams

Parameter type passed to interrupt handler functions.

```python
class InterruptParams(TypedDict):
    tool_call_name: str
    tool_call_args: Dict[str, Any]
    tool: BaseTool
```

**Field Descriptions:**

- `tool_call_name`: String type. Tool call name.
- `tool_call_args`: Dictionary mapping strings to any values. Tool call arguments.
- `tool`: BaseTool type. Tool instance.

## HumanInterruptHandler

Type alias for interrupt handler functions.

```python
HumanInterruptHandler = Callable[[InterruptParams], Any]
```
