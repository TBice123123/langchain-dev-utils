# Building Multi-Agents

## Overview

When building complex AI applications, multi-agent collaboration is a powerful architectural pattern. By assigning different responsibilities to specialized agents, you can achieve specialized division of labor and efficient collaboration.

There are various ways to implement multi-agent collaboration, among which **Tool Calling** is a common and flexible approach. By encapsulating subagents as tools, a master agent can dynamically delegate tasks to specialized subagents based on requirements.

This library provides two pre-built functions to simplify this approach:

| Function Name | Description |
|--------|----------|
| `wrap_agent_as_tool` | Wraps a single agent instance into an independent tool |
| `wrap_all_agents_as_tool` | Wraps multiple agent instances into a unified tool, specifying which subagent to call via parameters |

## Wrapping a Single Agent as a Tool

Wrapping a single agent takes just three steps:

1. Import `wrap_agent_as_tool`
2. Pass the agent instance as a parameter
3. Obtain a tool object that can be directly called by other agents

### Function Parameters

| Parameter | Description |
|------|------|
| `agent` | The agent instance, must have a `name` attribute defined.<br><br>**Type**: `CompiledStateGraph`<br>**Required**: Yes |
| `tool_name` | The tool name, defaults to `transfer_to_{agent_name}`.<br><br>**Type**: `str`<br>**Required**: No |
| `tool_description` | The tool description, defaults to `This tool transforms input to {agent_name}`.<br><br>**Type**: `str`<br>**Required**: No |
| `pre_input_hooks` | Hook functions before the agent runs.<br><br>**Type**: `tuple[Callable, Callable] | Callable`<br>**Required**: No |
| `post_output_hooks` | Hook functions after the agent runs.<br><br>**Type**: `tuple[Callable, Callable] | Callable`<br>**Required**: No |

### Usage Example

Below, we use the `supervisor` agent as an example to introduce how to wrap subagents as tools using `wrap_agent_as_tool`.

First, implement two subagents: one for sending emails and one for calendar querying and scheduling.

#### Email Agent
```python
from langchain_core.tools import tool
from langchain_dev_utils.chat_models import register_model_provider
from langchain_dev_utils.agents import create_agent, wrap_agent_as_tool 

register_model_provider(
    "vllm",
    "openai-compatible",
    base_url="http://localhost:8000/v1",
)


@tool
def send_email(
    to: list[str],  # Email addresses
    subject: str,
    body: str,
    cc: list[str] = [],
) -> str:
    """Send an email via the Email API. Requires correctly formatted addresses."""
    # Stub: In a real application, this would call SendGrid, Gmail API, etc.
    return f"Email sent to {', '.join(to)} - Subject: {subject}"


EMAIL_AGENT_PROMPT = (
    "You are an email assistant."
    "Draft professional emails based on natural language requests."
    "Extract recipient information and create appropriate subject lines and body content."
    "Use send_email to send the email."
    "Always confirm what was sent in the final reply."
)

email_agent = create_agent(
    "vllm:qwen3-4b",
    tools=[send_email],
    system_prompt=EMAIL_AGENT_PROMPT,
    name="email_agent",
)
```

#### Calendar Agent
```python
@tool
def create_calendar_event(
    title: str,
    start_time: str,  # ISO format: "2024-01-15T14:00:00"
    end_time: str,  # ISO format: "2024-01-15T15:00:00"
    attendees: list[str],  # Email addresses
    location: str = "",
) -> str:
    """Create a calendar event. Requires precise ISO date-time format."""
    # Stub: In a real application, this would call Google Calendar API, Outlook API, etc.
    return f"Event created: {title} from {start_time} to {end_time} with {len(attendees)} participants"


@tool
def get_available_time_slots(
    attendees: list[str],
    date: str,  # ISO format: "2024-01-15"
    duration_minutes: int,
) -> list[str]:
    """Query calendar availability for attendees on a specific date."""
    # Stub: In a real application, this would query the calendar API
    return ["09:00", "14:00", "16:00"]


CALENDAR_AGENT_PROMPT = (
    "You are a calendar scheduling assistant."
    "Parse natural language scheduling requests (e.g., 'next Tuesday at 2 PM') into the correct ISO date-time format."
    "Use get_available_time_slots to check availability when needed."
    "Use create_calendar_event to schedule events."
    "Always confirm what was scheduled in the final reply."
)

calendar_agent = create_agent(
    "vllm:qwen3-4b",
    tools=[create_calendar_event, get_available_time_slots],
    system_prompt=CALENDAR_AGENT_PROMPT,
    name="calendar_agent",
)
```

Next, use `wrap_agent_as_tool` to wrap these two subagents into tools.

```python
schedule_event = wrap_agent_as_tool(
    calendar_agent,
    tool_name="schedule_event",
    tool_description=(
        "Schedule calendar events using natural language."
        "Use this feature when the user wants to create, modify, or check calendar appointments."
        "Capable of handling date/time parsing, querying available times, and creating events."
        "Input: Natural language calendar scheduling request (e.g., 'Meeting with the design team next Tuesday at 2 PM')"
    ),
)
manage_email = wrap_agent_as_tool(
    email_agent,
    tool_name="manage_email",
    tool_description=(
        "Send emails using natural language."
        "Use this feature when the user wants to send notifications, reminders, or any email communication."
        "Capable of extracting recipient information, subject generation, and email drafting."
        "Input: Natural language email request (e.g., 'Send them a meeting reminder')"
    ),
)
```

Finally, create a `supervisor_agent` that can call these two tools.

```python
SUPERVISOR_PROMPT = (
    "You are a helpful personal assistant."
    "You can schedule calendar events and send emails."
    "Break down user requests into appropriate tool calls and coordinate the results."
    "When a request involves multiple operations, please use multiple tools sequentially."
)


supervisor_agent = create_agent(
    "vllm:qwen3-4b",
    tools=[schedule_event, manage_email],
    system_prompt=SUPERVISOR_PROMPT,
)

print(
    supervisor_agent.invoke({"messages": [HumanMessage(content="Check available time for tomorrow")]})
)
print(
    supervisor_agent.invoke(
        {"messages": [HumanMessage(content="Send a meeting reminder to test@123.com")]}
    )
)
```

!!! info "Tip"

    In the example above, we imported `create_agent` from `langchain_dev_utils.agents` instead of `langchain.agents`. This is because this library also provides a function with exactly the same functionality as the official `create_agent`, but with the added capability to specify models via strings. This allows you to directly use models registered via `register_model_provider` without needing to initialize and pass model instances.


## Wrapping Multiple Agents as a Single Tool

Wrapping multiple agents into a single tool takes just three steps:

1. Import `wrap_all_agents_as_tool`
2. Pass multiple agent instances as a list at once
3. Obtain a unified tool object that can be directly called by other agents

### Function Parameters

| Parameter | Description |
|------|------|
| `agents` | List of agent instances.<br><br>**Type**: `list[CompiledStateGraph]`<br>**Required**: Yes |
| `tool_name` | The tool name, defaults to `task`.<br><br>**Type**: `str`<br>**Required**: No |
| `tool_description` | The tool description, defaults to containing all available agent information.<br><br>**Type**: `str`<br>**Required**: No |
| `pre_input_hooks` | Hook functions before the agent runs.<br><br>**Type**: `tuple[Callable, Callable] | Callable`<br>**Required**: No |
| `post_output_hooks` | Hook functions after the agent runs.<br><br>**Type**: `tuple[Callable, Callable] | Callable`<br>**Required**: No |

### Usage Example

For the `calendar_agent` and `email_agent` from the previous example, we can wrap them into one tool `call_subagent`:

```python
call_subagent_tool = wrap_all_agents_as_tool(
    [calendar_agent, email_agent],
    tool_name="call_subagent",
    tool_description=(
        "Call a subagent to execute tasks."
        "Available agents include:"
        "- calendar_agent: for scheduling calendar events"
        "- email_agent: for sending emails"
    ),
)

MAIN_AGENT_PROMPT = (
    "You are a helpful personal assistant."
    "You can use the **call_subagent** tool to call subagents to execute tasks."
    "Break down user requests into appropriate tool calls and coordinate the results."
    "When a request involves multiple operations, please use multiple tools sequentially."
)

main_agent = create_agent(
    "vllm:qwen3-4b",
    tools=[call_subagent_tool],
    system_prompt=MAIN_AGENT_PROMPT,
)
```

!!! info "Tip"

    Besides using the `wrap_all_agents_as_tool` provided by this library to wrap multiple agents into a single tool, you can also use the `SubAgentMiddleware` provided by the `deepagents` library to achieve a similar effect.

## Hook Functions

This library includes a flexible hook mechanism that allows inserting custom logic before and after a subagent runs. This mechanism applies to both `wrap_agent_as_tool` and `wrap_all_agents_as_tool`. The following explanation uses `wrap_agent_as_tool` as an example.

### 1. pre_input_hooks

Pre-process the input before the agent runs. Useful for input augmentation, context injection, format validation, permission checks, etc.

#### Supported Input Types

| Type | Description |
|------|------|
| Single synchronous function | Used for both synchronous (`invoke`) and asynchronous (`ainvoke`) call paths (will not be awaited in the async path, called directly) |
| Tuple `(sync_func, async_func)` | The first function is for the synchronous call path; the second function (must be `async def`) is for the asynchronous call path and will be `await`ed |

#### Function Signature

```python
def pre_input_hook(request: str, runtime: ToolRuntime) -> str | dict[str, Any]:
    """
    Args:
        request: The original tool call input
        runtime: langchain's ToolRuntime

    Returns:
        The processed input, which serves as the actual input for the agent (needs to be str or dict)
    """
```

**Note**:

- The return value of the hook function must be str or dict, otherwise a ValueError will be raised.

- If returning dict, it will be used directly as the agent's actual input.

- If returning str, it will be wrapped as `HumanMessage(content=...)`, ultimately serving as the agent's actual input as `{"messages": [HumanMessage(content=...)]}`.

#### Usage Example

```python
def process_input(request: str, runtime: ToolRuntime) -> str:
    return "<task_description>" + request + "</task_description>"

# Or support async
async def process_input_async(request: str, runtime: ToolRuntime) -> str:
    return "<task_description>" + request + "</task_description>"

# Usage
call_agent_tool = wrap_agent_as_tool(
    agent,
    pre_input_hooks=(process_input, process_input_async)
)
```

!!! tip "Tip"

    The example above is relatively simple. In practice, you can add more complex logic based on the `state` or `context` within `runtime`.

### 2. post_output_hooks

Post-process the complete message list returned by the agent after it finishes running to generate the final return value of the tool. Useful for result extraction, structured transformation, etc.

#### Supported Input Types

| Type | Description |
|------|------|
| Single function | Used for both synchronous and asynchronous paths (will not be awaited in the async path) |
| Tuple `(sync_func, async_func)` | The first is for the synchronous path; the second (`async def`) is for the asynchronous path and will be `await`ed |

#### Function Signature

```python
def post_output_hook(request: str, response: dict[str, Any], runtime: ToolRuntime) -> Union[str, Command]:
    """
    Args:
        request: The unprocessed raw input
        response: The complete response returned by the agent
        runtime: langchain's ToolRuntime
    
    Returns:
        A value that can be serialized to a string, or a Command object
    """
```

#### Usage Example

```python
from langgraph.types import Command

def process_output_sync(request: str, response: dict[str, Any], runtime: ToolRuntime) -> Command:
    return Command(update={
        "messages":[ToolMessage(content=response["messages"][-1].content, tool_call_id=runtime.tool_call_id)]
    })

async def process_output_async(request: str, response: dict[str, Any], runtime: ToolRuntime) -> Command:
    return Command(update={
        "messages":[ToolMessage(content=response["messages"][-1].content, tool_call_id=runtime.tool_call_id)]
    })

# Usage
call_agent_tool = wrap_agent_as_tool(
    agent,
    post_output_hooks=(process_output_sync, process_output_async)
)
```

!!! tip "Tip"

    The example above is relatively simple. In practice, you can add more complex logic based on the `state` or `context` within `runtime`.

### Default Behavior

- If no `pre_input_hooks` are provided, the raw input is directly used as the agent's actual input as `{"messages": [HumanMessage(content=request)]}`.
- If no `post_output_hooks` are provided, it defaults to returning `response["messages"][-1].content` (i.e., the text content of the last message).