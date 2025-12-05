# 中间件

> [!NOTE]
>
> **功能概述**：提供方便进行 Agent 开发的实用工具。
>
> **前置要求**：了解 langchain 的[中间件](https://docs.langchain.com/oss/python/langchain/middleware)。
>
> **预计阅读时间**：12 分钟

## 概述

中间件是专门针对`langchain`预构建的 Agent 而构建的组件。官方提供了一些内置的中间件。本库则根据实际情况和本库的使用场景，提供了更多的中间件。

## 任务规划

任务规划的中间件，用于在执行复杂任务前进行结构化分解与过程管理。

::: tip 📝
任务规划是一种高效的上下文工程管理策略。在执行任务之前，大模型首先将整体任务拆解为多个有序的子任务，形成任务规划列表（在本库中称为 plan）。随后按顺序执行各子任务，并在每完成一个步骤后动态更新任务状态，直至所有子任务执行完毕。
:::

实现任务规划的中间件为`PlanMiddleware` ，其参数说明如下:

<Params
name="system_prompt"
type="string"
description="可选字符串类型，系统提示词。"
:required="false"
:default="null"
/>
<Params 
name="write_plan_tool_description"
type="string"
description="可选字符串类型，写计划工具的描述。"
:required="false"
:default="null"
/>
<Params 
name="finish_sub_plan_tool_description"
type="string"
description="可选字符串类型，完成子计划工具的描述。"
:required="false"
:default="null"
/>
<Params 
name="read_plan_tool_description"
type="string"
description="可选字符串类型，读计划工具的描述。"
:required="false"
:default="null"
/>
<Params 
name="use_read_plan_tool"
type="bool"
description="可选布尔类型，是否使用读计划工具。"
:required="false"
:default="true"
/>

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

这三个函数接收的参数如下:

<Params
name="description"
type="string"
description="工具描述,如果不传则采用默认的工具描述。"
:required="false"
:default="null"
/>
<Params
name="message_key"
type="string"
description="用于更新 messages 的键，若不传入则使用默认的 messages（read_plan 工具无此参数）。"
:required="false"
:default="null"
/>

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

<BestPractice>

一、使用 <code>create_agent</code> 时：

推荐直接使用 <code>PlanMiddleware</code>，而不是手动传入 <code>write_plan</code>、<code>finish_sub_plan</code>、<code>read_plan</code> 这三个工具。

原因：中间件已自动处理提示词构造和智能体状态管理，能显著降低使用复杂度。

注意：由于 <code>create_agent</code> 的模型输出固定更新到 <code>messages</code> 键，因此 <code>PlanMiddleware</code> 没有 <code>message_key</code> 参数。

二、使用 <code>langgraph</code> 时：

推荐直接使用这三个工具 (<code>write_plan</code>, <code>finish_sub_plan</code>, <code>read_plan</code>)。

原因：这种方式能更好地融入 <code>langgraph</code> 的自定义节点和状态管理。
</BestPractice>

## 模型路由

`ModelRouterMiddleware` 是一个用于**根据输入内容动态路由到最适配模型**的中间件。它通过一个“路由模型”分析用户请求，从预定义的模型列表中选择最适合当前任务的模型进行处理。

其参数如下:

<Params
name="router_model"
type="BaseChatModel | string"
description="用于执行路由决策的模型。可以传入字符串（将通过 load_chat_model 自动加载），例如 vllm:qwen3-4b；或直接传入已实例化的 BaseChatModel 对象。"
:required="true"
:default="null"
/>
<Params
name="model_list"
type="list[dict]"
description="一个模型配置列表，每个元素是一个字典，需包含 model_name (str), model_description (str)，以及可选的 tools (list[BaseTool]), model_kwargs (dict), model_system_prompt (str)。"
:required="true"
:default="null"
/>
<Params
name="router_prompt"
type="string"
description="自定义路由模型的提示词。若为 None（默认），则使用内置的默认提示模板。"
:required="false"
:default="null"
/>

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

::: info 工具权限配置  
`model_list` 中每个模型的工具权限，由其 `tools` 字段的配置决定，此配置遵循以下规则：

- **未定义时**：模型继承 `create_agent` 参数`tools`载入的全部工具。
- **定义为空列表 []**：模型被显式禁用所有工具。
- **定义为非空列表 [tool1, tool2, ...]**：此列表充当“工具白名单”，模型被严格限制仅能调用名单内的工具。所有在此指定的工具，必须已预先载入至 `create_agent `参数`tools`中。

:::

## 工具调用修复
`ToolCallRepairMiddleware` 是一个**自动修复大模型无效工具调用（`invalid_tool_calls`）**的中间件。

大模型在输出工具调用的 JSON Schema 时，可能因模型自身原因生成JSON格式错误的内容(错误的内容常见于`arguments` 字段)，导致 JSON 解析失败。这类调用会被存到 `invalid_tool_calls`字段中。`ToolCallRepairMiddleware` 会在模型返回结果后自动检测 `invalid_tool_calls`，并尝试调用 `json-repair` 进行修复，使工具调用得以正常执行。

请确保已安装 `langchain-dev-utils[standard]`，详见[安装指南](../installation.md)。

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

::: warning 注意
本中间件无法保证 100% 修复所有无效工具调用，实际效果取决于 `json-repair` 的修复能力；此外，它仅作用于 `invalid_tool_calls` 字段中的无效工具调用内容。
:::


::: info 注意
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
:::
