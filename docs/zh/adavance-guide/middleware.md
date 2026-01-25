# 中间件

## 概述

中间件是专门针对 LangChain 预构建的 Agent 而构建的组件。官方提供了一些内置的中间件，本库则根据实际使用场景提供了更多实用的中间件。

本库提供的中间件包括：

- `PlanMiddleware`：任务规划，将复杂任务拆解为有序子任务
- `ModelRouterMiddleware`：根据输入内容动态路由到最适配的模型
- `HandoffAgentMiddleware`：在多个子 Agent 之间灵活交接任务
- `ToolCallRepairMiddleware`：自动修复大模型无效工具调用
- `format_prompt`：动态格式化系统提示词中的占位符

此外，本库还扩充了官方中间件的功能，支持通过字符串参数指定模型：

- SummarizationMiddleware
- LLMToolSelectorMiddleware
- ModelFallbackMiddleware
- LLMToolEmulator

## 任务规划

`PlanMiddleware` 是一个用于在执行复杂任务前进行结构化分解与过程管理的中间件。

!!! info "补充说明"

    任务规划是一种高效的上下文工程管理策略。在执行任务之前，大模型首先将整体任务拆解为多个有序的子任务，形成任务规划列表（在本库中称为 plan）。随后按顺序执行各子任务，并在每完成一个步骤后动态更新任务状态，直至所有子任务执行完毕。

### 参数说明

| 参数 | 说明 |
|------|------|
| `system_prompt` | 系统提示词，若为 `None` 则使用默认提示词。<br><br>**类型**: `str`<br>**必填**: 否 |
| `custom_plan_tool_descriptions` | 自定义计划相关工具的描述。<br><br>**类型**: `dict`<br>**必填**: 否 |
| `use_read_plan_tool` | 是否启用读计划工具。<br><br>**类型**: `bool`<br>**必填**: 否<br>**默认值**: `True` |

`custom_plan_tool_descriptions` 字典的键可取以下三个值：

| 键 | 说明 |
|------|------|
| `write_plan` | 写计划工具的描述 |
| `finish_sub_plan` | 完成子计划工具的描述 |
| `read_plan` | 读计划工具的描述 |


### 使用示例

```python
from langchain_dev_utils.agents.middleware import PlanMiddleware

agent = create_agent(
    model="vllm:qwen3-4b",
    middleware=[
        PlanMiddleware(
            custom_plan_tool_descriptions={
                "write_plan": "用于写计划，将任务拆解为多个有序的子任务。",
                "finish_sub_plan": "用于完成子任务，更新子任务状态为已完成。",
                "read_plan": "用于查询当前的任务规划列表。"
            },
            use_read_plan_tool=True,  # 如果不使用读计划工具，可以设置此参数为 False
        )
    ],
)

response = agent.invoke(
    {"messages": [HumanMessage(content="我要去New York玩几天，帮我规划行程")]}
)
print(response)
```

### 工具说明

`PlanMiddleware` 要求必须使用 `write_plan` 和 `finish_sub_plan` 两个工具，而 `read_plan` 工具默认启用；若不需要使用，可将 `use_read_plan_tool` 参数设为 `False`。

### 与官方 To-do list 中间件的对比

本中间件与 LangChain 官方提供的 **To-do list 中间件** 功能定位相似，但在工具设计上存在差异：

| 特性 | 官方 To-do list 中间件 | 本库 PlanMiddleware |
|------|----------------------|---------------------|
| 工具数量 | 1 个（`write_todo`） | 3 个（`write_plan`、`finish_sub_plan`、`read_plan`） |
| 功能定位 | 面向待办清单（todo list） | 专门用于规划列表（plan list） |
| 操作方式 | 添加和修改通过一个工具完成 | 写入、修改、查询分别由不同工具完成 |

无论是 `todo` 还是 `plan`，其本质都是同一个概念。本中间件区别于官方的关键点在于提供了三个专用工具：

- `write_plan`：用于写入计划或更新计划内容
- `finish_sub_plan`：用于在完成某个子任务后更新其状态
- `read_plan`：用于查询计划内容

## 模型路由

`ModelRouterMiddleware` 是一个用于**根据输入内容动态路由到最适配模型**的中间件。它通过一个"路由模型"分析用户请求，从预定义的模型列表中选择最适合当前任务的模型进行处理。

### 参数说明

| 参数 | 说明 |
|------|------|
| `router_model` | 用于执行路由决策的模型。<br><br>**类型**: `str` \| `BaseChatModel`<br>**必填**: 是 |
| `model_list` | 模型配置列表。<br><br>**类型**: `list[ModelDict]`<br>**必填**: 是 |
| `router_prompt` | 自定义路由模型的提示词。<br><br>**类型**: `str`<br>**必填**: 否 |

#### `model_list` 配置说明

每个模型配置为一个字典，包含以下字段：

| 字段 | 说明 |
|------|------|
| `model_name` | 模型的唯一标识，使用 `provider:model-name` 格式。<br><br>**类型**: `str`<br>**必填**: 是 |
| `model_description` | 模型能力或适用场景的简要描述。<br><br>**类型**: `str`<br>**必填**: 是 |
| `tools` | 该模型可调用的工具白名单。<br><br>**类型**: `list[BaseTool]`<br>**必填**: 否 |
| `model_kwargs` | 模型加载时的额外参数。<br><br>**类型**: `dict`<br>**必填**: 否 |
| `model_system_prompt` | 模型的系统级提示词。<br><br>**类型**: `str`<br>**必填**: 否 |
| `model_instance` | 已实例化的模型对象。<br><br>**类型**: `BaseChatModel`<br>**必填**: 否 |


!!! tip "model_instance 字段说明"

    - **若提供**：直接使用该实例，`model_name` 仅作标识，`model_kwargs` 被忽略；适用于不使用本库的对话模型管理功能的情况。
    - **若未提供**：根据 `model_name` 和 `model_kwargs` 使用 `load_chat_model` 加载模型。
    - **命名格式**：无论哪种情况，`model_name` 的命名都推荐采用 `provider:model-name` 格式。


### 使用示例

#### 步骤一：定义模型列表

```python
from langchain_dev_utils.agents.middleware.model_router import ModelDict

model_list: list[ModelDict] = [
    {
        "model_name": "vllm:qwen3-8b",
        "model_description": "适合普通任务，如对话、文本生成等",
        "model_kwargs": {
            "temperature": 0.7,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
        "model_system_prompt": "你是一个助手，擅长处理普通任务，如对话、文本生成等。",
    },
    {
        "model_name": "vllm:qwen3-vl-2b",
        "model_description": "适合视觉任务",
        "tools": [],  # 如果该模型不需要任何工具，请将此字段设置为空列表 []
    },
    {
        "model_name": "vllm:qwen3-coder-flash",
        "model_description": "适合代码生成任务",
        "tools": [run_python_code],  # 仅允许使用 run_python_code 工具
    },
    {
        "model_name": "openai:gpt-4o",
        "model_description": "适合综合类高难度任务",
        "model_system_prompt": "你是一个助手，擅长处理综合类的高难度任务",
        "model_instance": ChatOpenAI(
            model_name="gpt-4o"
        ),  # 直接传入实例，此时 model_name 仅作标识，model_kwargs 被忽略
    },
]
```

#### 步骤二：创建 Agent 并启用中间件

```python
from langchain_dev_utils.agents.middleware import ModelRouterMiddleware
from langchain_core.messages import HumanMessage

agent = create_agent(
    model="vllm:qwen3-4b",  # 此模型仅作占位，实际由中间件动态替换
    tools=[run_python_code, get_current_time],
    middleware=[
        ModelRouterMiddleware(
            router_model="vllm:qwen3-4b",
            model_list=model_list,
        )
    ],
)

# 路由中间件会根据输入内容自动选择最合适的模型
response = agent.invoke({"messages": [HumanMessage(content="帮我写一个冒泡排序代码")]})
print(response)
```

通过 `ModelRouterMiddleware`，你可以轻松构建一个多模型、多能力的 Agent，根据任务类型自动选择最优模型，提升响应质量与效率。

!!! note "并行执行"
    采用中间件实现模型路由，每次仅会分配一个任务进行执行，如果你想要将任务分解为多个子任务由多个模型进行并行执行，请参考[状态图编排](pipeline.md)。

## 智能体交接

`HandoffAgentMiddleware` 是一个用于**在多个子 Agent 之间灵活切换**的中间件，完整实现了 LangChain 官方的 `handoffs` 多智能体协作方案。

### 参数说明

| 参数 | 说明 |
|------|------|
| `agents_config` | 智能体配置字典，键为智能体名称，值为智能体配置字典。<br><br>**类型**: `dict[str, AgentConfig]`<br>**必填**: 是 |
| `custom_handoffs_tool_descriptions` | 自定义交接工具的描述，键为智能体名称，值为对应的交接工具描述。<br><br>**类型**: `dict[str, str]`<br>**必填**: 否 |
| `handoffs_tool_overrides` | 自定义交接工具的实现，键为智能体名称，值为对应的交接工具实现。<br><br>**类型**: `dict[str, BaseTool]`<br>**必填**: 否 |

#### `agents_config` 配置说明

每个智能体配置为一个字典，包含以下字段：

| 字段 | 说明 |
|------|------|
| `model` | 指定该智能体使用的模型；若不传，则沿用 `create_agent` 的 `model` 参数对应的模型。支持字符串（须为 `provider:model-name` 格式，如 `vllm:qwen3-4b`）或 `BaseChatModel` 实例。<br><br>**类型**: `str` \| `BaseChatModel`<br>**必填**: 否 |
| `prompt` | 智能体的系统提示词。<br><br>**类型**: `str` \| `SystemMessage`<br>**必填**: 是 |
| `tools` | 智能体可调用的工具列表。<br><br>**类型**: `list[BaseTool]`<br>**必填**: 否 |
| `default` | 是否设为默认智能体；缺省为 `False`。全部配置中必须且只能有一个智能体设为 `True`。<br><br>**类型**: `bool`<br>**必填**: 否 |
| `handoffs` | 该智能体可交接给的其它智能体名称列表。若设为 `"all"`，则表示该智能体可交接给所有其它智能体。<br><br>**类型**: `list[str]` \| `str`<br>**必填**: 是 |

对于这种范式的多智能体实现，往往需要一个用于交接（handoffs）的工具。本中间件利用每个智能体的 `handoffs` 配置，自动为每个智能体创建对应的交接工具。如果要自定义交接工具的描述，则可以通过 `custom_handoffs_tool_descriptions` 参数实现。


**使用示例**

本示例中，将使用四个智能体：`time_agent`、`weather_agent`、`code_agent` 和 `default_agent`。

接下来要创建对应智能体的配置字典 `agent_config`。

```python
from langchain_dev_utils.agents.middleware.handoffs import AgentConfig

agent_config: dict[str, AgentConfig] = {
    "time_agent": {
        "model": "vllm:qwen3-8b",
        "prompt": "你是一个时间助手",
        "tools": [get_current_time],
        "handoffs": ["default_agent"],  # 该智能体只能交接到default_agent
    },
    "weather_agent": {
        "prompt": "你是一个天气助手",
        "tools": [get_current_weather, get_current_city],
        "handoffs": ["default_agent"],
    },
    "code_agent": {
        "model": load_chat_model("vllm:qwen3-coder-flash"),
        "prompt": "你是一个代码助手",
        "tools": [
            run_code,
        ],
        "handoffs": ["default_agent"],
    },
    "default_agent": {
        "prompt": "你是一个助手",
        "default": True, # 设为默认智能体
        "handoffs": "all",  # 该智能体可以交接到所有其它智能体
    },
}
```

最终将这个配置传递给 `HandoffAgentMiddleware` 即可。

```python
from langchain_dev_utils.agents.middleware import HandoffAgentMiddleware

agent = create_agent(
    model="vllm:qwen3-4b",
    tools=[
        get_current_time,
        get_current_weather,
        get_current_city,
        run_code,
    ],
    middleware=[HandoffAgentMiddleware(agents_config=agent_config)],
)

response = agent.invoke({"messages": [HumanMessage(content="当前时间是多少？")]})
print(response)
```

如果想要自定义交接工具的描述，可以传递第二个参数 `custom_handoffs_tool_descriptions`。

```python
agent = create_agent(
    model="vllm:qwen3-4b",
    tools=[
        get_current_time,
        get_current_weather,
        get_current_city,
        run_code,
    ],
    middleware=[
        HandoffAgentMiddleware(
            agents_config=agent_config,
            custom_handoffs_tool_descriptions={
                "time_agent": "此工具用于交接到时间助手去解决时间查询问题",
                "weather_agent": "此工具用于交接到天气助手去解决天气查询问题",
                "code_agent": "此工具用于交接到代码助手去解决代码问题",
                "default_agent": "此工具用于交接到默认的助手",
            },
        )
    ],
)
```

如果你想完全自定义实现交接工具的逻辑，则可以传递第三个参数 `handoffs_tool_overrides`。与第二个参数类似，它也是一个字典，键为智能体名称，值为对应的交接工具实现。

自定义交接工具必须返回一个 `Command` 对象，其 `update` 属性需包含 `messages` 键（返回工具响应）和 `active_agent` 键（值为要交接的智能体名称，用于切换当前智能体）。

例如：

```python
@tool
def transfer_to_code_agent(runtime: ToolRuntime) -> Command:
    """This tool help you transfer to the code agent."""
    #这里你可以添加自定义逻辑
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content="transfer to code agent",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "active_agent": "code_agent",
            #这里你可以添加其它的要更新的键
        }
    )

agent = create_agent(
    model="vllm:qwen3-4b",
    tools=[
        get_current_time,
        get_current_weather,
        get_current_city,
        run_code,
    ],
    middleware=[
        HandoffAgentMiddleware(
            agents_config=agent_config,
            handoffs_tool_overrides={
                "code_agent": transfer_to_code_agent,
            },
        )
    ],
)
```

`handoffs_tool_overrides` 用于高度定制化交接工具的实现，如果仅仅是想要自定义交接工具的描述，则应该使用 `custom_handoffs_tool_descriptions`。

## 工具调用修复

`ToolCallRepairMiddleware` 是一个**自动修复大模型无效工具调用（`invalid_tool_calls`）**的中间件。

大模型在输出工具调用的 JSON Schema 时，可能因模型自身原因生成 JSON 格式错误的内容（错误的内容常见于 `arguments` 字段），导致 JSON 解析失败。这类调用会被存到 `invalid_tool_calls` 字段中。`ToolCallRepairMiddleware` 会在模型返回结果后自动检测 `invalid_tool_calls`，并尝试调用 `json-repair` 进行修复，使工具调用得以正常执行。

请确保已安装 `langchain-dev-utils[standard]`，详见**安装指南**。

### 参数说明

该中间件零配置开箱即用，无需额外参数。

### 使用示例

```python
from langchain_dev_utils.agents.middleware import ToolCallRepairMiddleware

agent = create_agent(
    model="vllm:qwen3-4b",
    tools=[run_python_code, get_current_time],
    middleware=[
        ToolCallRepairMiddleware()
    ],
)
```

!!! warning "注意"
    本中间件无法保证 100% 修复所有无效工具调用，实际效果取决于 `json-repair` 的修复能力；此外，它仅作用于 `invalid_tool_calls` 字段中的无效工具调用内容。


## 格式化系统提示词

`format_prompt` 是一个**中间件实例**，允许您在 `system_prompt` 中使用 `f-string` 风格的占位符（如 `{name}`），并在运行时动态地用实际值替换它们。

### 参数说明

占位符中的变量值遵循一个明确的解析顺序：

1. **优先从 `state` 中查找**：会先从 `state` 字典中查找与占位符同名的字段
2. **其次从 `context` 中查找**：如果在 `state` 中未找到该字段，则会继续在 `context` 对象中查找

这个顺序意味着 `state` 中的值拥有更高的优先级，可以覆盖 `context` 中同名的值。

### 使用示例

#### 仅从 `state` 中获取变量

这是最基础的用法，所有占位符变量都由 `state` 提供。

```python
from langchain_dev_utils.agents.middleware import format_prompt
from langchain.agents import AgentState

class AssistantState(AgentState):
    name: str

agent = create_agent(
    model="vllm:qwen3-4b",
    system_prompt="你是一个智能助手，你的名字叫做{name}。",
    middleware=[format_prompt],
    state_schema=AssistantState,
)

# 在调用时，必须为 state 提供 'name' 的值
response = agent.invoke(
    {"messages": [HumanMessage(content="你好啊")], "name": "assistant"}
)
print(response)
```

#### 同时从 `state` 和 `context` 中获取变量

同时使用 `state` 和 `context`：

```python
from dataclasses import dataclass

@dataclass
class Context:
    user: str

agent = create_agent(
    model="vllm:qwen3-4b",
    # {name} 将从 state 获取，{user} 将从 context 获取
    system_prompt="你是一个智能助手，你的名字叫做{name}。你的使用者叫做{user}。",
    middleware=[format_prompt],
    state_schema=AssistantState,
    context_schema=Context,
)

# 在调用时，为 state 提供 'name'，为 context 提供 'user'
response = agent.invoke(
    {
        "messages": [HumanMessage(content="我要去New York玩几天，帮我规划行程")],
        "name": "assistant",
    },
    context=Context(user="张三"),
)
print(response)
```

#### 变量覆盖示例

此示例展示了当 `state` 和 `context` 中存在同名变量时，`state` 的值会优先生效。

```python
from dataclasses import dataclass

@dataclass
class Context:
    # context 中定义了 'name'
    name: str
    user: str

agent = create_agent(
    model="vllm:qwen3-4b",
    system_prompt="你是一个智能助手，你的名字叫做{name}。你的使用者叫做{user}。",
    middleware=[format_prompt],
    state_schema=AssistantState, # state 中也定义了 'name'
    context_schema=Context,
)

# 在调用时，state 和 context 都提供了 'name' 的值
response = agent.invoke(
    {
        "messages": [HumanMessage(content="你叫什么名字？")],
        "name": "assistant-1",
    },
    context=Context(name="assistant-2", user="张三"),
)

# 最终的系统提示词会是 "你是一个智能助手，你的名字叫做assistant-1。你的使用者叫做张三。"
# 因为 state 的优先级更高
print(response)
```

!!! warning "注意"
    自定义中间件有两种实现方式：装饰器或继承类。  
    - 继承类实现：`PlanMiddleware`、`ModelMiddleware`、`HandoffAgentMiddleware`、`ToolCallRepairMiddleware`  
    - 装饰器实现：`format_prompt`（装饰器会把函数直接变成中间件实例，因此无需手动实例化即可使用）


!!! info "官方中间件扩充"
    本库扩充了以下官方中间件，支持通过字符串参数指定已被 `register_model_provider` 注册的模型：

    你只需要导入本库中的这些中间件，即可使用字符串指定已经被`register_model_provider`注册的模型。中间件使用方法和官方中间件保持一致，例如：
    ```python
    from langchain_core.messages import AIMessage
    from langchain_dev_utils.agents.middleware import SummarizationMiddleware
    from langchain_dev_utils.chat_models import register_model_provider

    register_model_provider(
        provider_name="vllm",
        chat_model="openai-compatible",
        base_url="http://localhost:8000/v1",
    )
    agent = create_agent(
        model="vllm:qwen3-4b",
        middleware=[
            SummarizationMiddleware(
                model="vllm:qwen3-4b",
                trigger=("tokens", 50),
                keep=("messages", 1),
            )
        ],
        system_prompt="你是一个智能的AI助手，可以解决用户的问题",
    )
    response = agent.invoke({"messages": messages})
    print(response)
    ```