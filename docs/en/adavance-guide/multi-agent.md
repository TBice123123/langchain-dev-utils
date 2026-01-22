# Subagent Tools (Agent as Tool)

## Overview

When building complex AI applications, multi-agent collaboration is a powerful architectural pattern. By assigning different responsibilities to specialized agents, you can achieve specialized division of labor and efficient collaboration.

There are various ways to implement multi-agent collaboration, among which **Tool Calling** is a common and flexible approach. By encapsulating subagents as tools, a master agent can dynamically delegate tasks to specialized subagents based on requirements.

This library provides two pre-built functions to simplify this implementation:

| Function Name | Description |
|---------------|-------------|
| `wrap_agent_as_tool` | Encapsulates a single agent instance as an independent tool |
| `wrap_all_agents_as_tool` | Encapsulates multiple agent instances into a unified tool, specifying which subagent to call via parameters |

## Wrapping a Single Agent as a Tool

Wrapping a single agent involves just three steps:

1. Import `wrap_agent_as_tool`
2. Pass the agent instance as a parameter
3. Obtain a tool object that can be directly invoked by other agents

### Function Parameter Description

| Parameter | Description |
|-----------|-------------|
| `agent` | Agent instance, must have a defined `name` attribute.<br><br>**Type**: `CompiledStateGraph`<br>**Required**: Yes |
| `tool_name` | Tool name, defaults to `transfer_to_{agent_name}`.<br><br>**Type**: `str`<br>**Required**: No |
| `tool_description` | Tool description, defaults to `This tool transforms input to {agent_name}`.<br><br>**Type**: `str`<br>**Required**: No |
| `pre_input_hooks` | Hook functions before the agent runs.<br><br>**Type**: `tuple[Callable, Callable] | Callable`<br>**Required**: No |
| `post_output_hooks` | Hook functions after the agent runs.<br><br>**Type**: `tuple[Callable, Callable] | Callable`<br>**Required**: No |

### Usage Example

Below, we use the `supervisor` agent as an example to demonstrate how to wrap a subagent as a tool using `wrap_agent_as_tool`.

First, implement two subagents: one for sending emails and one for calendar queries and scheduling.

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
    """Sends an email via the email API. Requires correctly formatted addresses."""
    # Stub: In a real application, you would call SendGrid, Gmail API, etc. here
    return f"Email sent to {', '.join(to)} - Subject: {subject}"


EMAIL_AGENT_PROMPT = (
    "You are an email assistant. "
    "Draft professional emails based on natural language requests. "
    "Extract recipient information and create appropriate subject lines and body content. "
    "Use send_email to send emails. "
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
    """Creates a calendar event. Requires precise ISO date-time format."""
    # Stub: In a real application, you would call Google Calendar API, Outlook API, etc. here
    return f"Event created: {title} from {start_time} to {end_time} with {len(attendees)} participants"


@tool
def get_available_time_slots(
    attendees: list[str],
    date: str,  # ISO format: "2024-01-15"
    duration_minutes: int,
) -> list[str]:
    """Queries calendar availability for attendees on a specific date."""
    # Stub: In a real application, you would query the calendar API here
    return ["09:00", "14:00", "16:00"]


CALENDAR_AGENT_PROMPT = (
    "You are a calendar scheduling assistant. "
    "Parse natural language scheduling requests (e.g., 'next Tuesday at 2 PM') into the correct ISO date-time format. "
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

Next, use `wrap_agent_as_tool` to wrap these two subagents as tools.

```python
schedule_event = wrap_agent_as_tool(
    calendar_agent,
    tool_name="schedule_event",
    tool_description=(
        "Schedule calendar events using natural language. "
        "Use this when the user wants to create, modify, or check calendar appointments. "
        "Capable of handling date/time parsing, querying available times, and creating events. "
        "Input: Natural language scheduling request (e.g., 'meeting with the design team next Tuesday at 2 PM')"
    ),
)
manage_email = wrap_agent_as_tool(
    email_agent,
    tool_name="manage_email",
    tool_description=(
        "Send emails using natural language. "
        "Use this when the user wants to send notifications, reminders, or any email communication. "
        "Capable of extracting recipient information, subject generation, and email drafting. "
        "Input: Natural language email request (e.g., 'send them a meeting reminder')"
    ),
)
```

Finally, create a `supervisor_agent` that can invoke these two tools.

```python
SUPERVISOR_PROMPT = (
    "You are a helpful personal assistant. "
    "You can schedule calendar events and send emails. "
    "Break down user requests into appropriate tool calls and coordinate results. "
    "When a request involves multiple operations, please use multiple tools in sequence."
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

!!! info "Hint"

    In the example above, we imported `create_agent` from `langchain_dev_utils.agents` instead of `langchain.agents`. This is because this library also provides a function identical in functionality to the official `create_agent`, but with the added capability to specify models via strings. This allows you to directly use models registered via `register_model_provider` without needing to initialize model instances first.

## Wrapping Multiple Agents as a Single Tool

Wrapping multiple agents as a single tool involves just three steps:

1. Import `wrap_all_agents_as_tool`
2. Pass multiple agent instances as a list at once
3. Obtain a unified tool object that can be directly invoked by other agents

### Function Parameter Description

| Parameter | Description |
|-----------|-------------|
| `agents` | List of agent instances.<br><br>**Type**: `list[CompiledStateGraph]`<br>**Required**: Yes |
| `tool_name` | Tool name, defaults to `task`.<br><br>**Type**: `str`<br>**Required**: No |
| `tool_description` | Tool description, defaults to including all available agent information.<br><br>**Type**: `str`<br>**Required**: No |
| `pre_input_hooks` | Hook functions before the agent runs.<br><br>**Type**: `tuple[Callable, Callable] | Callable`<br>**Required**: No |
| `post_output_hooks` | Hook functions after the agent runs.<br><br>**Type**: `tuple[Callable, Callable] | Callable`<br>**Required**: No |

### Usage Example

For the `calendar_agent` and `email_agent` from the previous example, we can wrap them into a single tool `call_subagent`:

```python
call_subagent_tool = wrap_all_agents_as_tool(
    [calendar_agent, email_agent],
    tool_name="call_subagent",
    tool_description=(
        "Call subagents to execute tasks. "
        "Available agents include: "
        "- calendar_agent: for scheduling calendar events "
        "- email_agent: for sending emails"
    ),
)

MAIN_AGENT_PROMPT = (
    "You are a helpful personal assistant. "
    "You can use the **call_subagent** tool to call subagents to execute tasks. "
    "Break down user requests into appropriate tool calls and coordinate results. "
    "When a request involves multiple operations, please use multiple tools in sequence."
)

main_agent = create_agent(
    "vllm:qwen3-4b",
    tools=[call_subagent_tool],
    system_prompt=MAIN_AGENT_PROMPT,
)
```

!!! info "Hint"

    In addition to using `wrap_all_agents_as_tool` provided by this library to wrap multiple agents into a single tool, you can also achieve similar effects using the `SubAgentMiddleware` middleware provided by the `deepagents` library.

## Hook Functions

This library includes a flexible hook mechanism that allows you to insert custom logic before and after a subagent runs. This mechanism applies to both `wrap_agent_as_tool` and `wrap_all_agents_as_tool`. The following description uses `wrap_agent_as_tool` as an example.

### 1. pre_input_hooks

Preprocesses the input before the agent runs. Useful for input enhancement, context injection, format validation, permission checks, etc.

#### Supported Input Types

| Type | Description |
|------|-------------|
| Single sync function | Used for both sync (`invoke`) and async (`ainvoke`) call paths (will not be awaited in the async path, called directly) |
| Tuple `(sync_func, async_func)` | The first function is used for the sync call path; the second function (must be `async def`) is used for the async call path and will be `awaited` |

#### Function Signature

```python
def pre_input_hook(request: str, runtime: ToolRuntime) -> str | dict[str, Any]:
    """
    Args:
        request: The original tool call input
        runtime: langchain's ToolRuntime

    Returns:
        The processed input, which serves as the actual input for the agent (must be str or dict)
    """
```

**Note**:

- The return value of the hook function must be a `str` or `dict`; otherwise, a `ValueError` will be raised.

- If a `dict` is returned, it will be used directly as the agent's actual input.

- If a `str` is returned, it will be wrapped as `HumanMessage(content=...)`, ultimately serving as the agent's actual input in the format `{"messages": [HumanMessage(content=...)]}`.

- If `pre_input_hooks` is not provided, the original input is used directly as the agent's actual input in the format `{"messages": [HumanMessage(content=request)]}`.

#### Usage Example

For instance, before calling a subagent, you can use a model to summarize the master agent's conversation history, providing more precise task context for the subagent.

```python
from langchain.tools import ToolRuntime
from langchain_core.messages import SystemMessage
from langchain_dev_utils.agents import wrap_agent_as_tool


def process_input(request: str, runtime: ToolRuntime) -> str:
    messages = runtime.state.get("messages", [])

    new_messages = [
        SystemMessage(
            content="""Please generate a concise and accurate summary based on the following conversation history.
            The summary should include:
            1) Topic of the conversation;
            2) Key content;
            3) Current status or progress.
            Keep the summary length under 200 characters."""
        ),
        *messages,
    ]

    if messages:
        summary = model.invoke(new_messages)

        return (
            "<history_summary>\n"
            + summary.content
            + "\n</history_summary>\n"
            + "<task_description>\n"
            + request
            + "\n</task_description>"
        )
    return "<task_description>\n" + request + "\n</task_description>"


async def process_input_async(request: str, runtime: ToolRuntime) -> str:
    messages = runtime.state.get("messages", [])

    new_messages = [
        SystemMessage(
            content="""Please generate a concise and accurate summary based on the following conversation history.
            The summary should include:
            1) Topic of the conversation;
            2) Key information points;
            3) Current status or progress.
            Keep the summary length under 200 characters."""
        ),
        *messages,
    ]

    if messages:
        summary = await model.ainvoke(new_messages)

        return (
            "<history_summary>\n"
            + summary.content
            + "\n</history_summary>\n"
            + "<task_description>\n"
            + request
            + "\n</task_description>"
        )
    return "<task_description>\n" + request + "\n</task_description>"


# Usage
call_agent_tool = wrap_agent_as_tool(
    agent, pre_input_hooks=(process_input, process_input_async)
)
```


### 2. post_output_hooks

Performs post-processing on the complete list of messages returned by the agent after it has finished running, to generate the final return value of the tool. Useful for result extraction, structured transformation, etc.

#### Supported Input Types

| Type | Description |
|------|-------------|
| Single function | Used for both sync and async paths (will not be awaited in async path) |
| Tuple `(sync_func, async_func)` | The first is used for the sync path; the second (`async def`) is used for the async path and will be `awaited` |

#### Function Signature

```python
def post_output_hook(request: str, response: dict[str, Any], runtime: ToolRuntime) -> Union[str, Command]:
    """
    Args:
        request: The unprocessed original input
        response: The complete response returned by the agent
        runtime: langchain's ToolRuntime
    
    Returns:
        A value that can be serialized to a string, or a Command object
    """
```

**Note**:

- The return value of the hook function must be a value that can be serialized to a string or a `Command` object.

- The two input arguments of the hook function are: `request`, which is the unprocessed original input, and `response`, which is the complete response returned by the agent (i.e., the return value of `agent.invoke(input)`).

- If `post_output_hooks` is not provided, the agent's final response is used directly as the tool's return value (i.e., `response["messages"][-1].content`).

#### Usage Example

For example, after the subagent finishes execution, besides updating the `messages` key, it might update other state keys. If you need to save these additional state keys to the master agent's state, you can use a `Command` object for the return.

```python
from typing import Any
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langgraph.types import Command


def process_output_sync(
    request: str, response: dict[str, Any], runtime: ToolRuntime
) -> Command:
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=response["messages"][-1].content,
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "example_state_key": response["example_state_key"],
        }
    )


async def process_output_async(
    request: str, response: dict[str, Any], runtime: ToolRuntime
) -> Command:
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=response["messages"][-1].content,
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "example_state_key": response["example_state_key"],
        }
    )


# Usage
call_agent_tool = wrap_agent_as_tool(
    agent, post_output_hooks=(process_output_sync, process_output_async)
)
```