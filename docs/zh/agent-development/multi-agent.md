# 多智能体构建

> [!NOTE]
>
> **功能概述**：提供方便进行Multi-Agent开发的实用工具。
>
> **前置要求**：了解 langchain 的[Agent](https://docs.langchain.com/oss/python/langchain/agents)、[Multi-Agent](https://docs.langchain.com/oss/python/langchain/multi-agent)。
>
> **预计阅读时间**：7 分钟

## 概述

将智能体封装为工具是多智能体系统中的一种常见实现模式，LangChain 官方文档对此有详细阐述。为此，本库提供了预构建函数 `wrap_agent_as_tool` 来实现此模式，该函数能够将一个智能体实例封装成一个可供其它智能体调用的工具。


其参数如下:

<Params
name="agent"
type="CompiledStateGraph"
description="智能体，取值必须为 langgraph 的 CompiledStateGraph。"
:required="true"
:default="null"
/>
<Params
name="tool_name"
type="string"
description="工具的名称。如果不传则工具默认名称是`transfor_to_agent_name`。"
:required="false"
:default="null"
/>
<Params
name="tool_description"
type="string"
description="工具的描述。如果不传则使用默认的描述内容。"
:required="false"
:default="null"
/>
<Params
name="pre_input_hooks"
type="Callable | tuple[Callable, AwaitableCallable]"
description="预处理钩子函数，可以是单个同步函数或一个二元组。如果是二元组，则第一个函数是同步函数，第二个函数是异步函数，用于在智能体运行前对输入进行预处理。"
:required="false"
:default="null"
/>
<Params
name="post_output_hooks"
type="Callable | tuple[Callable, AwaitableCallable]"
description="后处理钩子函数，可以是单个同步函数或一个二元组。如果是二元组，则第一个函数是同步函数，第二个函数是异步函数，用于在智能体运行完成后，对其返回的完整消息列表进行后处理。"
:required="false"
:default="null"
/>



## 使用示例

下面，我们以官方示例中的 `supervisor` 智能体为基础，介绍如何通过 `wrap_agent_as_tool` 将其快速改造成可被其它智能体调用的工具。

首先实现两个子智能体，一个用于发送邮件，一个用于日程查询和安排。

**邮件智能体**
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
    to: list[str],  # 电子邮件地址
    subject: str,
    body: str,
    cc: list[str] = [],
) -> str:
    """通过电子邮件API发送邮件。要求正确格式的地址。"""
    # 存根：实际应用中，这里会调用SendGrid、Gmail API等
    return f"邮件已发送至 {', '.join(to)} - 主题: {subject}"


EMAIL_AGENT_PROMPT = (
    "你是一个电子邮件助手。"
    "根据自然语言请求撰写专业邮件。"
    "提取收件人信息并制作恰当的主题行和正文内容。"
    "使用 send_email 来发送邮件。"
    "始终在最终回复中确认已发送的内容。"
)

email_agent = create_agent(
    "vllm:qwen3-4b",
    tools=[send_email],
    system_prompt=EMAIL_AGENT_PROMPT,
    name="email_agent",
)
```

**日程智能体**
```python
@tool
def create_calendar_event(
    title: str,
    start_time: str,  # ISO格式: "2024-01-15T14:00:00"
    end_time: str,  # ISO格式: "2024-01-15T15:00:00"
    attendees: list[str],  # 电子邮件地址
    location: str = "",
) -> str:
    """创建日历事件。要求精确的ISO日期时间格式。"""
    # 存根：实际应用中，这里会调用Google Calendar API、Outlook API等
    return f"事件已创建：{title} 从 {start_time} 到 {end_time}，共有 {len(attendees)} 位参与者"


@tool
def get_available_time_slots(
    attendees: list[str],
    date: str,  # ISO格式: "2024-01-15"
    duration_minutes: int,
) -> list[str]:
    """在特定日期查询参与者的日历可用时间。"""
    # 存根：实际应用中，这里会查询日历API
    return ["09:00", "14:00", "16:00"]


CALENDAR_AGENT_PROMPT = (
    "你是一个日历日程安排助手。"
    "将自然语言的日程安排请求（例如'下周二下午2点'）解析为正确的ISO日期时间格式。"
    "需要时使用 get_available_time_slots 来检查可用时间。"
    "使用 create_calendar_event 来安排事件。"
    "始终在最终回复中确认已安排的内容。"
)

calendar_agent = create_agent(
    "vllm:qwen3-4b",
    tools=[create_calendar_event, get_available_time_slots],
    system_prompt=CALENDAR_AGENT_PROMPT,
    name="calendar_agent",
)
```

接下来，使用`wrap_agent_as_tool`将该两个子智能体封装为工具。

```python
schedule_event = wrap_agent_as_tool(
    calendar_agent,
    "schedule_event",
    tool_description="""使用自然语言安排日历事件。

    在用户想要创建、修改或检查日历约会时使用此功能。
    能够处理日期/时间解析、查询可用时间和创建事件。

    输入：自然语言日历安排请求（例如'与设计团队下个星期二下午2点的会议'）
    """,
)
manage_email = wrap_agent_as_tool(
    email_agent,
    "manage_email",
    tool_description="""使用自然语言发送电子邮件。

    在用户想要发送通知、提醒或任何电子邮件通信时使用此功能。
    能够提取收件人信息、主题生成和电子邮件撰写。

    输入：自然语言电子邮件请求（例如'向他们发送会议提醒'）
    """,
)
```

最终创建一个`supervisor_agent`，它可以调用这两个工具。

```python
SUPERVISOR_PROMPT = (
    "你是一个有用的个人助手。"
    "你可以安排日历事件并发送电子邮件。"
    "将用户请求分解为适当的工具调用，并协调结果。"
    "当请求涉及多个操作时，请使用多个工具按顺序操作。"
)


supervisor_agent = create_agent(
    "vllm:qwen3-4b",
    tools=[schedule_event, manage_email],
    system_prompt=SUPERVISOR_PROMPT,
)
```

测试一下功能：

```python
print(
    supervisor_agent.invoke({"messages": [HumanMessage(content="查询明天的空闲时间")]})
)
print(
    supervisor_agent.invoke(
        {"messages": [HumanMessage(content="给test@123.com发送邮件会议提醒")]}
    )
)
```

::: info 注意
上述示例中，我们是从`langchain_dev_utils.agents`中导入了`create_agent`函数，而不是`langchain.agents`，这是因为本库也提供了一个与官方`create_agent`函数功能完全相同的函数，只是扩充了通过字符串指定模型的功能。使得可以直接使用`register_model_provider`注册的模型，而无需初始化模型实例后传入。
:::

## 钩子函数

本函数提供了几个钩子函数，用于在调用智能体前后进行一些操作。

#### 1. pre_input_hooks

在智能体运行前对输入进行预处理。可用于输入增强、上下文注入、格式校验、权限检查等。

支持传入以下类型：

- 若传入 **单个同步函数**，则该函数同时用于同步（`invoke`）和异步（`ainvoke`）调用路径（异步路径中不会 `await`，直接调用）。
- 若传入 **二元组 `(sync_func, async_func)`**：
  - 第一个函数用于同步调用路径；
  - 第二个函数（必须是 `async def`）用于异步调用路径，并会被 `await`。

你传入的函数接收两个参数：

- `request: str`：原始工具调用输入；
- `runtime: ToolRuntime`：`langchain`的`ToolRuntime`。

你传入的函数必须返回处理后的 `str`，作为 agent 的实际输入。

**示例**：

```python
def process_input(request: str, runtime: ToolRuntime) -> str:
    return "<task_description>" + request + "</task_description>"

# 或支持异步
async def process_input_async(request: str, runtime: ToolRuntime) -> str:
    return "<task_description>" + request + "</task_description>"

# 使用
call_agent_tool = wrap_agent_as_tool(
    agent,
    pre_input_hooks=(process_input, process_input_async)
)
```

注意，上述的例子比较简单，实际上你可以根据`runtime`里面的`state`或者`context`添加更复杂的逻辑。

#### 2. post_output_hooks

在智能体运行完成后，对其返回的完整消息列表进行后处理，以生成工具的最终返回值。可用于结果提取、结构化转换等。

支持传入以下类型：

- 若传入 **单个函数**，该函数用于同步和异步路径（异步路径中不 `await`）。
- 若传入 **二元组 `(sync_func, async_func)`**：
  - 第一个用于同步路径；
  - 第二个（`async def`）用于异步路径，并会被 `await`。

你传入的函数接收三个参数：

- `request: str`：（可能已处理的）原始输入；
- `messages: List[AnyMessage]`：agent 返回的完整消息历史（来自 `response["messages"]`）；
- `runtime: ToolRuntime`：`langchain`的`ToolRuntime`。

你传入的函数返回的值可以是能够被序列化为一个字符串或者是`Command`对象。

**示例**：

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

# 使用
call_agent_tool = wrap_agent_as_tool(
    agent,
    post_output_hooks=(process_output_sync, process_output_async)
)
```

- 若未提供 `pre_input_hooks`，输入原样传递；
- 若未提供 `post_output_hooks`，默认返回 `response["messages"][-1].content`（即最后一条消息的文本内容）。

注意，上述的例子比较简单，实际上你可以根据`runtime`里面的`state`或者`context`添加更复杂的逻辑。


**注意**：当该 Agent（CompiledStateGraph）作为 `wrap_agent_as_tool` 的 agent 参数时，该 Agent 必须定义 name 属性。
