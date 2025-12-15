# 中间件


## 概述

中间件是专门针对`langchain`预构建的 Agent 而构建的组件。官方提供了一些内置的中间件。本库则根据实际情况和本库的使用场景，提供了更多的中间件。

## 任务规划

任务规划的中间件，用于在执行复杂任务前进行结构化分解与过程管理。

!!! note "补充"
    任务规划是一种高效的上下文工程管理策略。在执行任务之前，大模型首先将整体任务拆解为多个有序的子任务，形成任务规划列表（在本库中称为 plan）。随后按顺序执行各子任务，并在每完成一个步骤后动态更新任务状态，直至所有子任务执行完毕。

实现任务规划的中间件为`PlanMiddleware`，其中接收以下参数：

- `system_prompt`：可选字符串类型，系统提示词。默认值为 `None`，将使用默认的系统提示词。
- `write_plan_tool_description`：可选字符串类型，写计划工具的描述。默认值为 `None`，将使用默认的写计划工具描述。
- `finish_sub_plan_tool_description`：可选字符串类型，完成子计划工具的描述。默认值为 `None`，将使用默认的完成子计划工具描述。
- `read_plan_tool_description`：可选字符串类型，读计划工具的描述。默认值为 `None`，将使用默认的读计划工具描述。
- `use_read_plan_tool`：可选布尔类型，是否使用读计划工具。默认值为 `True`，将启用读计划工具。


**使用示例**：

```python
from langchain_dev_utils.agents.middleware import PlanMiddleware

agent = create_agent(
    model="vllm:qwen3-4b",
    middleware=[
        PlanMiddleware(
            use_read_plan_tool=True, #如果不使用读计划工具，可以设置此参数为False
        )
    ],
)

response = agent.invoke(
    {"messages": [HumanMessage(content="我要去New York玩几天，帮我规划行程")]}
)
print(response)
```

`PlanMiddleware` 要求必须使用 `write_plan` 和 `finish_sub_plan` 两个工具，而 `read_plan` 工具默认启用；若不需要使用，可将 `use_read_plan_tool` 参数设为 `False`。

本中间件与 LangChain 官方提供的 **To-do list 中间件**功能定位相似，但在工具设计上存在差异。官方中间件仅提供 `write_todo` 工具，面向的是待办清单（todo list）结构；而本库则提供了 `write_plan` 、`finish_sub_plan`、`read_plan` 三个专用工具，专门用于对规划列表（plan list）进行写入、修改、查询等操作。

无论是`todo`还是`plan`其本质都是同一个，因此本中间件区别于官方的关键点在于提供的工具，官方的添加和修改是通过一个工具来完成的，而本库则提供了三个工具，其中`write_plan`可用于写入计划或者更新计划内容，`finish_sub_plan`则用于在完成某个子任务后更新其状态，`read_plan`用于查询计划内容。

同时，本库还提供了三个函数来创建上述这三个工具:

- `create_write_plan_tool`：创建一个用于写计划的工具的函数
- `create_finish_sub_plan_tool`：创建一个用于完成子任务的工具的函数
- `create_read_plan_tool`：创建一个用于查询计划的工具的函数

这三个函数都可以接收一个`description`参数,用于自定义工具的描述。如果不传入,则采用默认的工具描述。其中`create_write_plan_tool`和`create_finish_sub_plan_tool`还可以接收一个`message_key`参数,用于自定义更新 messages 的键。如果不传入,则采用默认的`messages`键。

**使用示例**：

```python
from langchain_dev_utils.agents.middleware.plan import (
    create_write_plan_tool,
    create_finish_sub_plan_tool,
    create_read_plan_tool,
    PlanState,
)

agent = create_agent(
    model="vllm:qwen3-4b",
    state_schema=PlanState,
    tools=[create_write_plan_tool(), create_finish_sub_plan_tool(), create_read_plan_tool()],
)
```

需要注意的是,要使用这三个工具,你必须要保证状态 Schema 中包含 plan 这个键,否则会报错,对此你可以使用本库提供的`PlanState`来继承状态 Schema。

!!! note "最佳实践"
    一、使用 `create_agent` 时：

    推荐直接使用 `PlanMiddleware`，而不是手动传入 `write_plan`、`finish_sub_plan`、`read_plan` 这三个工具。

    原因：中间件已自动处理提示词构造和智能体状态管理，能显著降低使用复杂度。

    注意：由于 `create_agent` 的模型输出固定更新到 `messages` 键，因此 `PlanMiddleware` 没有 `message_key` 参数。

    二、使用 `langgraph` 时：

    推荐直接使用这三个工具 (`write_plan`, `finish_sub_plan`, `read_plan`)。

    原因：这种方式能更好地融入 `langgraph` 的自定义节点和状态管理。


## 模型路由

`ModelRouterMiddleware` 是一个用于**根据输入内容动态路由到最适配模型**的中间件。它通过一个“路由模型”分析用户请求，从预定义的模型列表中选择最适合当前任务的模型进行处理。

其参数如下：

- `router_model`：用于执行路由决策的模型。可以传入字符串（将通过 `load_chat_model` 自动加载），例如 `vllm:qwen3-4b`；或直接传入已实例化的 `BaseChatModel` 对象。
- `model_list`：模型配置列表，每个元素为一个字典，其中可以包含以下字段：
    - `model_name`（str）：必传，模型的唯一标识，**使用 `provider:model-name` 格式**，例如 `vllm:qwen3-4b` 或 `openrouter:qwen/qwen3-vl-32b-instruct`；
    - `model_description`（str）：必传，模型能力或适用场景的简要描述，供路由模型进行决策。
    - `tools`（list[BaseTool]）：可选，该模型可调用的工具白名单。  
            - 若未提供，则继承全局工具列表；  
            - 若设为 `[]`，则显式禁用所有工具。
    - `model_kwargs`（dict）：可选，模型加载时的额外参数（如 `temperature`、`max_tokens` 等），**仅在未传入 `model_instance` 时生效**。
    - `model_instance`（BaseChatModel）：可选，已实例化的模型对象。  
            - 若提供，则直接使用该实例，`model_name` 仅作标识，**不再通过 `load_chat_model` 加载**，且 `model_kwargs` 被忽略；  
            - 若未提供，则会根据 `model_name` 和 `model_kwargs` 自动加载模型。
    - `model_system_prompt`（str）：可选，模型的系统级提示词。
- `router_prompt`：自定义路由模型的提示词。若为 `None`（默认），则使用内置的默认提示模板。


**使用示例**

首先定义模型列表：

```python
model_list = [
    {
        "model_name": "vllm:qwen3-8b",
        "model_description": "适合普通任务，如对话、文本生成等",
        "model_kwargs": {
            "temperature": 0.7,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
        },
        "model_system_prompt": "你是一个助手，擅长处理普通任务，如对话、文本生成等。",
    },
    {
        "model_name": "openrouter:qwen/qwen3-vl-32b-instruct",
        "model_description": "适合视觉任务",
        "tools": [],  # 如果该模型不需要任何工具，请将此字段设置为空列表 []
    },
    {
        "model_name": "openrouter:qwen/qwen3-coder-plus",
        "model_description": "适合代码生成任务",
        "tools": [run_python_code],  # 仅允许使用 run_python_code 工具
    },
]
```


然后在创建 agent 时启用中间件：

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



## 工具调用修复
`ToolCallRepairMiddleware` 是一个**自动修复大模型无效工具调用（`invalid_tool_calls`）**的中间件。

大模型在输出工具调用的 JSON Schema 时，可能因模型自身原因生成JSON格式错误的内容(错误的内容常见于`arguments` 字段)，导致 JSON 解析失败。这类调用会被存到 `invalid_tool_calls`字段中。`ToolCallRepairMiddleware` 会在模型返回结果后自动检测 `invalid_tool_calls`，并尝试调用 `json-repair` 进行修复，使工具调用得以正常执行。

请确保已安装 `langchain-dev-utils[standard]`，详见**安装指南**。

该中间件零配置开箱即用，无需额外参数。

**使用示例：**

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

本中间件 `format_prompt` 允许您在 `system_prompt` 中使用 `f-string` 风格的占位符（如 `{name}`），并在运行时动态地用实际值替换它们。

占位符中的变量值遵循一个明确的解析顺序：

1.  **优先从 `state` 中查找**：会先从`state`字典中查找与占位符同名的字段。
2.  **其次从 `context` 中查找**：如果在 `state` 中未找到该字段，则会继续在 `context` 对象中查找。

这个顺序意味着 `state` 中的值拥有更高的优先级，可以覆盖 `context` 中同名的值。

使用示例如下：

- **仅从 `state` 中获取变量**

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

- **同时从 `state` 和 `context` 中获取变量**

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

- **变量覆盖示例**

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
    - 继承类实现：`PlanMiddleware`、`ModelMiddleware`、`ToolCallRepairMiddleware`  
    - 装饰器实现：`format_prompt`（装饰器会把函数直接变成中间件实例，因此无需手动实例化即可使用）


!!! info "注意"
    除此之外，本库还扩充了以下中间件通过字符串参数指定模型的功能：

    - SummarizationMiddleware

    - LLMToolSelectorMiddleware

    - ModelFallbackMiddleware
    
    - LLMToolEmulator

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
    # big_text 是一个包含大量内容的文本，这里省略
    big_messages = [
        HumanMessage(content="你好，你是谁"),
        AIMessage(content="我是你的AI助手"),
        HumanMessage(content="写一段优美的长文本"),
        AIMessage(content=f"好的，我会写一段优美的长文本，内容是：{big_text}"),
        HumanMessage(content="你为啥要写这段长文本呢？"),
    ]
    response = agent.invoke({"messages": big_messages})
    print(response)
    ```