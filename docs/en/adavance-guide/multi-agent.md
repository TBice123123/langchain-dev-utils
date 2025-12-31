# Multi-Agent Construction

## Overview

When building complex AI applications, multi-agent collaboration is a powerful architectural pattern. By assigning different responsibilities to specialized agents, you can achieve task specialization and efficient collaboration.

There are multiple ways to implement multi-agent collaboration, with **tool calling** being a common and flexible implementation approach. By encapsulating sub-agents as tools, a main agent can dynamically delegate tasks to specialized sub-agents based on requirements.

This library provides two pre-built functions to simplify this implementation:

| Function Name | Description |
|--------------|-------------|
| `wrap_agent_as_tool` | Wraps a single agent instance as an independent tool |
| `wrap_all_agents_as_tool` | Wraps multiple agent instances as a unified tool, specifying which sub-agent to call through parameters |

## Wrapping a Single Agent as a Tool

Wrapping a single agent requires just three steps:

1. Import `wrap_agent_as_tool`
2. Pass the agent instance as a parameter
3. Obtain a tool object that can be directly called by other agents

### Function Parameter Description

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `agent` | `CompiledStateGraph` | Yes | Agent instance, must have a defined `name` attribute |
| `tool_name` | `str` | No | Tool name, defaults to `transfer_to_{agent_name}` |
| `tool_description` | `str` | No | Tool description, defaults to `This tool transforms input to {agent_name}` |
| `pre_input_hooks` | `tuple` | No | Hook functions to run before the agent executes |
| `post_output_hooks` | `tuple` | No | Hook functions to run after the agent executes |

### Usage Example

Below, we'll use the `supervisor` agent as an example to demonstrate how to wrap sub-agents as tools using `wrap_agent_as_tool`.

First, let's implement two sub-agents: one for sending emails and another for calendar queries and scheduling.

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
    """Send an email through the email API. Requires properly formatted addresses."""
    # Stub: In a real application, this would call SendGrid, Gmail API, etc.
    return f"Email sent to {', '.join(to)} - Subject: {subject}"


EMAIL_AGENT_PROMPT = (
    "You are an email assistant. "
    "Compose professional emails based on natural language requests. "
    "Extract recipient information and create appropriate subject lines and body content. "
    "Use send_email to send the email. "
    "Always confirm what was sent in your final response."
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
    # Stub: In a real application, this would query calendar APIs
    return ["09:00", "14:00", "16:00"]


CALENDAR_AGENT_PROMPT = (
    "You are a calendar scheduling assistant. "
    "Parse natural language scheduling requests (e.g., 'next Tuesday at 2 PM') into proper ISO date-time formats. "
    "Use get_available_time_slots to check availability when needed. "
    "Use create_calendar_event to schedule events. "
    "Always confirm what was scheduled in your final response."
)

calendar_agent = create_agent(
    "vllm:qwen3-4b",
    tools=[create_calendar_event, get_available_time_slots],
    system_prompt=CALENDAR_AGENT_PROMPT,
    name="calendar_agent",
)
```

Next, use `wrap_agent_as_tool` to wrap these two sub-agents as tools.

```python
schedule_event = wrap_agent_as_tool(
    calendar_agent,
    tool_name="schedule_event",
    tool_description=(
        "Schedule calendar events using natural language. "
        "Use this when users want to create, modify, or check calendar appointments. "
        "Capable of handling date/time parsing, querying availability, and creating events. "
        "Input: Natural language calendar scheduling request (e.g., 'Meeting with design team next Tuesday at 2 PM')"
    ),
)
manage_email = wrap_agent_as_tool(
    email_agent,
    tool_name="manage_email",
    tool_description=(
        "Send emails using natural language. "
        "Use this when users want to send notifications, reminders, or any email communication. "
        "Capable of extracting recipient information, subject generation, and email composition. "
        "Input: Natural language email request (e.g., 'Send them a meeting reminder')"
    ),
)
```

Finally, create a `supervisor_agent` that can call these two tools.

```python
SUPERVISOR_PROMPT = (
    "You are a helpful personal assistant. "
    "You can schedule calendar events and send emails. "
    "Break down user requests into appropriate tool calls and coordinate the results. "
    "When a request involves multiple operations, use multiple tools sequentially."
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

    In the example above, we imported the `create_agent` function from `langchain_dev_utils.agents` instead of `langchain.agents`. This is because this library also provides a function with exactly the same functionality as the official `create_agent` function, but with the added ability to specify models through strings. This allows direct use of models registered with `register_model_provider` without needing to initialize model instances first.


## Wrapping Multiple Agents as a Single Tool

Wrapping multiple agents as a single tool requires just three steps:

1. Import `wrap_all_agents_as_tool`
2. Pass multiple agent instances as a list at once
3. Obtain a unified tool object that can be directly called by other agents

### Function Parameter Description

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `agents` | `list[CompiledStateGraph]` | Yes | List of agent instances |
| `tool_name` | `str` | No | Tool name, defaults to `task` |
| `tool_description` | `str` | No | Tool description, defaults to include information about all available agents |
| `pre_input_hooks` | `tuple` | No | Hook functions to run before agents execute |
| `post_output_hooks` | `tuple` | No | Hook functions to run after agents execute |

### Usage Example

For the `calendar_agent` and `email_agent` from the previous example, we can wrap them as a single tool `call_subagent`:

```python
call_subagent_tool = wrap_all_agents_as_tool(
    [calendar_agent, email_agent],
    tool_name="call_subagent",
    tool_description=(
        "Call sub-agents to perform tasks. "
        "Available agents: "
        "- calendar_agent: For scheduling calendar events"
        "- email_agent: For sending emails"
    ),
)

MAIN_AGENT_PROMPT = (
    "You are a helpful personal assistant. "
    "You can use the **call_subagent** tool to call sub-agents to perform tasks. "
    "Break down user requests into appropriate tool calls and coordinate the results. "
    "When a request involves multiple operations, use multiple tools sequentially."
)

main_agent = create_agent(
    "vllm:qwen3-4b",
    tools=[call_subagent_tool],
    system_prompt=MAIN_AGENT_PROMPT,
)
```

!!! info "Tip"

    Besides using the `wrap_all_agents_as_tool` provided by this library to wrap multiple agents as a single tool, you can also use the `SubAgentMiddleware` middleware provided by the `deepagents` library to achieve similar effects.

## Hook Functions

This library has a built-in flexible hook mechanism that allows inserting custom logic before and after sub-agent execution. This mechanism applies to both `wrap_agent_as_tool` and `wrap_all_agents_as_tool`. The following section uses `wrap_agent_as_tool` as an example for explanation.

### 1. pre_input_hooks

Preprocess the input before the agent runs. Can be used for input enhancement, context injection, format validation, permission checking, etc.

#### Supported Input Types

| Type | Description |
|------|-------------|
| Single synchronous function | Used for both synchronous (`invoke`) and asynchronous (`ainvoke`) call paths (will not be `awaited` in the async path, called directly) |
| Binary tuple `(sync_func, async_func)` | The first function is used for the synchronous call path; the second function (must be `async def`) is used for the asynchronous call path and will be `awaited` |

#### Function Signature

```python
def pre_input_hook(request: str, runtime: ToolRuntime) -> str:
    """
    Parameters:
        request: Original tool call input
        runtime: LangChain's ToolRuntime
    
    Returns:
        Processed str, as the actual input to the agent
    """
```

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

    The above example is relatively simple. In practice, you can add more complex logic based on the `state` or `context` within `runtime`.

### 2. post_output_hooks

After the agent completes execution, post-process the complete message list it returns to generate the final return value of the tool. Can be used for result extraction, structured transformation, etc.

#### Supported Input Types

| Type | Description |
|------|-------------|
| Single function | Used for both synchronous and asynchronous paths (will not be `awaited` in the async path) |
| Binary tuple `(sync_func, async_func)` | The first is used for the synchronous path; the second (`async def`) is used for the asynchronous path and will be `awaited` |

#### Function Signature

```python
def post_output_hook(request: str, messages: list, runtime: ToolRuntime) -> Union[str, Command]:
    """
    Parameters:
        request: (Possibly processed) original input
        messages: Complete message history returned by the agent (from response["messages"])
        runtime: LangChain's ToolRuntime
    
    Returns:
        A value that can be serialized to a string, or a Command object
    """
```

#### Usage Example

```python
from langgraph.types import Command

def process_output_sync(request: str, messages: list, runtime: ToolRuntime) -> Command:
    return Command(update={
        "messages":[ToolMessage(content=messages[-1].content, tool_call_id=runtime.tool_call_id)]
    })

async def process_output_async(request: str, messages: list, runtime: ToolRuntime) -> Command:
    return Command(update={
        "messages":[ToolMessage(content=messages[-1].content, tool_call_id=runtime.tool_call_id)]
    })

# Usage
call_agent_tool = wrap_agent_as_tool(
    agent,
    post_output_hooks=(process_output_sync, process_output_async)
)
```

!!! tip "Tip"

    The above example is relatively simple. In practice, you can add more complex logic based on the `state` or `context` within `runtime`.

### Default Behavior

- If `pre_input_hooks` is not provided, the input is passed through as is
- If `post_output_hooks` is not provided, it defaults to returning `response["messages"][-1].content` (i.e., the text content of the last message)