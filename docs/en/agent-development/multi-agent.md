# Multi-Agent Construction

> [!NOTE]
>
> **Function Overview**: Provides practical utilities for facilitating Multi-Agent development.
>
> **Prerequisites**: Understanding of LangChain's [Agent](https://docs.langchain.com/oss/python/langchain/agents) and [Multi-Agent](https://docs.langchain.com/oss/python/langchain/multi-agent).
>
> **Estimated Reading Time**: 7 minutes

## Overview

Encapsulating agents as tools is a common implementation pattern in multi-agent systems, which is elaborated in the official LangChain documentation. To this end, this library provides a pre-built function `wrap_agent_as_tool` to implement this pattern, which can encapsulate an agent instance into a tool that can be called by other agents.

Its parameters are as follows:

<Params
name="agent"
type="CompiledStateGraph"
description="The agent, which must be a langgraph CompiledStateGraph."
:required="true"
:default="null"
/>
<Params
name="tool_name"
type="string"
description="The name of the tool. If not provided, the default name is `transfor_to_agent_name`."
:required="false"
:default="null"
/>
<Params
name="tool_description"
type="string"
description="The description of the tool. If not provided, the default description is used."
:required="false"
:default="null"
/>
<Params
name="pre_input_hooks"
type="Callable | tuple[Callable, AwaitableCallable]"
description="Preprocessing hook function, which can be a single synchronous function or a tuple. If it's a tuple, the first function is synchronous, and the second is asynchronous, used for preprocessing input before the agent runs."
:required="false"
:default="null"
/>
<Params
name="post_output_hooks"
type="Callable | tuple[Callable, AwaitableCallable]"
description="Post-processing hook function, which can be a single synchronous function or a tuple. If it's a tuple, the first function is synchronous, and the second is asynchronous, used for post-processing the complete message list returned by the agent after it completes."
:required="false"
:default="null"
/>

## Usage Example

Below, we'll use the `supervisor` agent from the official example to introduce how to quickly transform it into a tool that can be called by other agents using `wrap_agent_as_tool`.

First, implement two sub-agents: one for sending emails and one for calendar query and scheduling.

**Email Agent**
```python{3}
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
    """Send emails through email API. Requires properly formatted addresses."""
    # Stub: In actual applications, this would call SendGrid, Gmail API, etc.
    return f"Email sent to {', '.join(to)} - Subject: {subject}"


EMAIL_AGENT_PROMPT = (
    "You are an email assistant."
    "Compose professional emails based on natural language requests."
    "Extract recipient information and create appropriate subject lines and body content."
    "Use send_email to send emails."
    "Always confirm what was sent in your final response."
)

email_agent = create_agent(
    "vllm:qwen3-4b",
    tools=[send_email],
    system_prompt=EMAIL_AGENT_PROMPT,
    name="email_agent",
)
```

**Calendar Agent**
```python
@tool
def create_calendar_event(
    title: str,
    start_time: str,  # ISO format: "2024-01-15T14:00:00"
    end_time: str,  # ISO format: "2024-01-15T15:00:00"
    attendees: list[str],  # Email addresses
    location: str = "",
) -> str:
    """Create calendar events. Requires precise ISO date-time format."""
    # Stub: In actual applications, this would call Google Calendar API, Outlook API, etc.
    return f"Event created: {title} from {start_time} to {end_time} with {len(attendees)} participants"


@tool
def get_available_time_slots(
    attendees: list[str],
    date: str,  # ISO format: "2024-01-15"
    duration_minutes: int,
) -> list[str]:
    """Query calendar availability for attendees on a specific date."""
    # Stub: In actual applications, this would query calendar APIs
    return ["09:00", "14:00", "16:00"]


CALENDAR_AGENT_PROMPT = (
    "You are a calendar scheduling assistant."
    "Parse natural language scheduling requests (e.g., 'next Tuesday at 2 PM') into correct ISO date-time format."
    "Use get_available_time_slots to check availability when needed."
    "Use create_calendar_event to schedule events."
    "Always confirm what was scheduled in your final response."
)

calendar_agent = create_agent(
    "vllm:qwen3-4b",
    tools=[create_calendar_event, get_available_time_slots],
    system_prompt=CALENDAR_AGENT_PROMPT,
    name="calendar_agent",
)
```

Next, use `wrap_agent_as_tool` to encapsulate these two sub-agents as tools.

```python
schedule_event = wrap_agent_as_tool(
    calendar_agent,
    "schedule_event",
    tool_description="""Schedule calendar events using natural language.

    Use this function when users want to create, modify, or check calendar appointments.
    Capable of handling date/time parsing, querying available time, and creating events.

    Input: Natural language scheduling request (e.g., 'Meeting with the design team next Tuesday at 2 PM')
    """,
)
manage_email = wrap_agent_as_tool(
    email_agent,
    "manage_email",
    tool_description="""Send emails using natural language.

    Use this function when users want to send notifications, reminders, or any email communication.
    Capable of extracting recipient information, subject generation, and email composition.

    Input: Natural language email request (e.g., 'Send them a meeting reminder')
    """,
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
```

Test the functionality:

```python
print(
    supervisor_agent.invoke({"messages": [HumanMessage(content="Query available time for tomorrow")]})
)
print(
    supervisor_agent.invoke(
        {"messages": [HumanMessage(content="Send a meeting reminder to test@123.com")]}
    )
)
```

::: info Note
In the example above, we imported the `create_agent` function from `langchain_dev_utils.agents` instead of `langchain.agents`. This is because this library provides a function identical to the official `create_agent`, but extended with the ability to specify models via strings. This allows you to use models registered via `register_model_provider` directly, without needing to initialize a model instance first.
:::


## Hook Functions

This function provides several hook functions for performing operations before and after calling the agent.

#### 1. pre_input_hooks

Preprocess the input before the agent runs. Can be used for input enhancement, context injection, format validation, permission checks, etc.

Supports the following types:

- If a **single synchronous function** is passed, it will be used for both synchronous (`invoke`) and asynchronous (`ainvoke`) call paths (it will be called directly without `await` in the asynchronous path).
- If a **tuple `(sync_func, async_func)`** is passed:
  - The first function is used for the synchronous call path;
  - The second function (must be `async def`) is used for the asynchronous call path and will be `await`ed.

The function you pass receives two parameters:

- `request: str`: The original tool call input;
- `runtime: ToolRuntime`: The `ToolRuntime` of `langchain`.

The function you pass must return the processed `str` as the actual input to the agent.

**Example**:

```python
def process_input(request: str, runtime: ToolRuntime) -> str:
    return "<task_description>" + request + "</task_description>"

# Or support asynchronous
async def process_input_async(request: str, runtime: ToolRuntime) -> str:
    return "<task_description>" + request + "</task_description>"

# Usage
call_agent_tool = wrap_agent_as_tool(
    agent,
    pre_input_hooks=(process_input, process_input_async)
)
```

Note that the above example is relatively simple. In practice, you can add more complex logic based on the `state` or `context` in `runtime`.

#### 2. post_output_hooks

Post-process the complete message list returned by the agent after it completes to generate the final return value of the tool. Can be used for result extraction, structured transformation, etc.

Supports the following types:

- If a **single function** is passed, it will be used for both synchronous and asynchronous paths (it will not be `await`ed in the asynchronous path).
- If a **tuple `(sync_func, async_func)`** is passed:
  - The first is used for the synchronous path;
  - The second (must be `async def`) is used for the asynchronous path and will be `await`ed.

The function you pass receives three parameters:

- `request: str`: The original input (possibly processed);
- `messages: List[AnyMessage]`: The complete message history returned by the agent (from `response["messages"]`);
- `runtime: ToolRuntime`: The `ToolRuntime` of `langchain`.

The value returned by the function you pass can be serialized to a string or a `Command` object.

**Example**:

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

- If `pre_input_hooks` is not provided, the input is passed as is;
- If `post_output_hooks` is not provided, it defaults to returning `response["messages"][-1].content` (i.e., the text content of the last message).

Note that the above example is relatively simple. In practice, you can add more complex logic based on the `state` or `context` in `runtime`.

**Note**: When this Agent (CompiledStateGraph) is used as the agent parameter for `wrap_agent_as_tool`, the Agent must define the name attribute.