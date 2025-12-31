# 多智能体构建


## 概述

将智能体封装为工具是多智能体系统中的一种实现模式，LangChain 官方文档将其称为`subagents`模式。该模式通过将子智能体（subagents）封装为工具，使主智能体能够根据任务需求动态委派给专门的子智能体处理，从而实现任务的专业化分工和协作。

本库提供了两个预构建函数来实现此模式：

- `wrap_agent_as_tool`：将单个智能体实例封装为一个独立工具

- `wrap_all_agents_as_tool`：将多个智能体实例封装为一个统一工具，通过参数指定调用哪个子智能体

## 封装单个智能体为工具

封装单个智能体只需三步：  
1. 导入 `wrap_agent_as_tool`  
2. 把智能体实例作为参数传入  
3. 获得可直接被其他智能体调用的工具对象  

函数第一个参数 `agent` 为必填项，需传入一个 `CompiledStateGraph` 实例；同时，该实例必须已定义 `name` 属性。  
可选参数 `tool_name` 与 `tool_description` 可分别指定工具的名称与描述；若留空，工具名默认取 `transfer_to_{agent_name}`，描述默认为 `This tool transforms input to {agent_name}`。  
此外，还可额外传入钩子函数，在智能体执行前后插入自定义逻辑。

### 使用示例

下面，我们以官方示例中的 `supervisor` 智能体为基础，介绍如何通过 `wrap_agent_as_tool` 将其快速改造成可被其它智能体调用的工具。

首先实现两个子智能体，一个用于发送邮件，一个用于日程查询和安排。

**邮件智能体**
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
    tool_name="schedule_event",
    tool_description=(
        "使用自然语言安排日历事件。"
        "在用户想要创建、修改或检查日历约会时使用此功能。"
        "能够处理日期/时间解析、查询可用时间和创建事件。"
        "输入：自然语言日历安排请求（例如'与设计团队下个星期二下午2点的会议'）"
    ),
)
manage_email = wrap_agent_as_tool(
    email_agent,
    tool_name="manage_email",
    tool_description=(
        "使用自然语言发送电子邮件。"
        "在用户想要发送通知、提醒或任何电子邮件通信时使用此功能。"
        "能够提取收件人信息、主题生成和电子邮件撰写。"
        "输入：自然语言电子邮件请求（例如'向他们发送会议提醒'）"
    ),
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

print(
    supervisor_agent.invoke({"messages": [HumanMessage(content="查询明天的空闲时间")]})
)
print(
    supervisor_agent.invoke(
        {"messages": [HumanMessage(content="给test@123.com发送邮件会议提醒")]}
    )
)
```

!!! info "注意"
    上述示例中，我们是从`langchain_dev_utils.agents`中导入了`create_agent`函数，而不是`langchain.agents`，这是因为本库也提供了一个与官方`create_agent`函数功能完全相同的函数，只是扩充了通过字符串指定模型的功能。使得可以直接使用`register_model_provider`注册的模型，而无需初始化模型实例后传入。


## 封装多个智能体封装为单一工具
将多个智能体封装为单一工具只需三步：  
1. 导入 `wrap_all_agents_as_tool`  
2. 把多个智能体实例作为列表一次性传入  
3. 获得可直接被其他智能体调用的统一工具对象  

函数第一个参数 `agents` 必须提供；可选参数 `tool_name` 与 `tool_description` 可自定义工具名称与描述。若省略，工具名默认为 `task`，描述默认为 `Launch an ephemeral subagent for a task.\nAvailable agents:\n {all_available_agents}`。  

同样支持也支持传入钩子函数，用于在智能体调用前或后执行自定义逻辑。

### 使用示例

对于上一个示例的`calendar_agent`和`email_agent`，我们可以将它们封装为一个工具`call_subagent`

```python
call_subagent_tool = wrap_all_agents_as_tool(
    [calendar_agent, email_agent],
    tool_name="call_subagent",
    tool_description=(
        "调用子智能体执行任务。"
        "可以使用的智能体有："
        "- calendar_agent：用于安排日历事件"
        "- email_agent：用于发送电子邮件"
    ),
)

MAIN_AGENT_PROMPT = (
    "你是一个有用的个人助手。"
    "你可以使用**call_subagent**工具调用子智能体执行任务。"
    "将用户请求分解为适当的工具调用，并协调结果。"
    "当请求涉及多个操作时，请使用多个工具按顺序操作。"
)

main_agent = create_agent(
    "vllm:qwen3-4b",
    tools=[call_subagent_tool],
    system_prompt=MAIN_AGENT_PROMPT,
)
```

!!! info "注意"
    除了使用本库提供的`wrap_all_agents_as_tool`将多个智能体封装为单一工具外，你还可以使用`deepagents`库提供的`SubAgentMiddleware`中间件实现类似的效果

## 钩子函数

本库内置了灵活的钩子（hook）机制，允许在子智能体运行前后插入自定义逻辑。  
该机制同时适用于 `wrap_agent_as_tool` 与 `wrap_all_agents_as_tool`，下文以 `wrap_agent_as_tool` 为例进行说明。

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


