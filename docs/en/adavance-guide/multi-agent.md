# Multi-Agent Construction


## Overview

Wrapping agents as tools is a common implementation pattern in multi-agent systems, which is detailed in the official LangChain documentation. To facilitate this pattern, this library provides a pre-built function `wrap_agent_as_tool` that can encapsulate an agent instance into a tool that can be called by other agents.

## Usage Example

Below, we'll use the `supervisor` agent from the official example as a basis to demonstrate how to quickly transform it into a tool that can be called by other agents using `wrap_agent_as_tool`.

First, let's implement two sub-agents: one for sending emails and another for calendar queries and scheduling.

**Email Agent**
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
    """Send emails via email API. Requires properly formatted addresses."""
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
    "Parse natural language scheduling requests (e.g., 'next Tuesday at 2 PM') into correct ISO date-time format. "
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
    tool_description="""Schedule calendar events using natural language.

    Use this function when users want to create, modify, or check calendar appointments.
    Capable of handling date/time parsing, querying available times, and creating events.

    Input: Natural language calendar scheduling request (e.g., 'Meeting with the design team next Tuesday at 2 PM')
    """,
)
manage_email = wrap_agent_as_tool(
    email_agent,
    tool_name="manage_email",
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
    "You are a helpful personal assistant. "
    "You can schedule calendar events and send emails. "
    "Break down user requests into appropriate tool calls and coordinate the results. "
    "When requests involve multiple operations, please use multiple tools in sequence."
)


supervisor_agent = create_agent(
    "vllm:qwen3-4b",
    tools=[schedule_event, manage_email],
    system_prompt=SUPERVISOR_PROMPT,
)
```

Let's test the functionality:

```python
print(
    supervisor_agent.invoke({"messages": [HumanMessage(content="Check available time slots for tomorrow")]})
)
print(
    supervisor_agent.invoke(
        {"messages": [HumanMessage(content="Send a meeting reminder to test@123.com")]}
    )
)
```

!!! info "Note"
    In the example above, we imported the `create_agent` function from `langchain_dev_utils.agents` instead of `langchain.agents`. This is because this library also provides a function with exactly the same functionality as the official `create_agent` function, but with the added capability to specify models through strings. This allows direct use of models registered with `register_model_provider` without needing to initialize model instances before passing them.


## Hook Functions

This function provides several hook functions for performing operations before and after calling the agent.

#### 1. pre_input_hooks

Preprocess the input before the agent runs. Can be used for input enhancement, context injection, format validation, permission checks, etc.

Supports passing the following types:

- If passing a **single synchronous function**, that function will be used for both synchronous (`invoke`) and asynchronous (`ainvoke`) call paths (the function will not be `await`ed in the asynchronous path, but called directly).
- If passing a **tuple `(sync_func, async_func)`**:
  - The first function is used for the synchronous call path;
  - The second function (must be `async def`) is used for the asynchronous call path and will be `await`ed.

The function you pass receives two parameters:

- `request: str`: The original tool call input;
- `runtime: ToolRuntime`: The `ToolRuntime` from `langchain`.

The function you pass must return a processed `str` as the actual input to the agent.

**Example**:

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

Note that the above example is relatively simple. In practice, you can add more complex logic based on the `state` or `context` within `runtime`.

#### 2. post_output_hooks

After the agent has finished running, post-process its complete message list to generate the final return value of the tool. Can be used for result extraction, structured transformation, etc.

Supports passing the following types:

- If passing a **single function**, that function is used for both synchronous and asynchronous paths (the function will not be `await`ed in the asynchronous path).
- If passing a **tuple `(sync_func, async_func)`**:
  - The first is used for the synchronous path;
  - The second (must be `async def`) is used for the asynchronous path and will be `await`ed.

The function you pass receives three parameters:

- `request: str`: The original input (possibly already processed);
- `messages: List[AnyMessage]`: The complete message history returned by the agent (from `response["messages"]`);
- `runtime: ToolRuntime`: The `ToolRuntime` from `langchain`.

The value returned by the function you pass can be something that can be serialized to a string or a `Command` object.

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

Note that the above example is relatively simple. In practice, you can add more complex logic based on the `state` or `context` within `runtime`.


**Note**: When an Agent (CompiledStateGraph) is used as the agent parameter of `wrap_agent_as_tool`, that Agent must define the name attribute.