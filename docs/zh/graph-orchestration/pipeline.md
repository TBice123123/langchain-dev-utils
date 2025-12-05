# 状态图编排

> [!NOTE]
>
> **功能概述**：主要用于实现多个状态图的并行和串行组合。
>
> **前置要求**：了解 langchain 的[子图](https://docs.langchain.com/oss/python/langgraph/use-subgraphs),[Send](https://docs.langchain.com/oss/python/langgraph/graph-api#send)。
>
> **预计阅读时间**：8 分钟

## 概述

提供方便进行状态图编排的实用工具。主要包含以下功能：

- 将多个状态图按照顺序方式进行编排，形成顺序工作流。
- 将多个状态图按照并行方式进行编排，形成并行工作流。

## 顺序编排

即用于搭建智能体顺序工作流（Sequential Pipeline）。这是一种将复杂任务分解为一系列连续、有序的子任务，并交由不同的专门化智能体依次处理的工作模式。

通过下面函数实现:

- `create_sequential_pipeline` - 以顺序方式组合多个状态图

其参数如下:
<Params
name="sub_graphs"
type="list[StateGraph | CompiledStateGraph]"
description="要组合的状态图列表（必须是 StateGraph 或者 CompiledStateGraph 实例）"
:required="true"
:default="null"
/>
<Params
name="state_schema"
type="dict"
description="最终生成图的 State Schema。"
:required="true"
:default="null"
/>
<Params
name="graph_name"
type="string"
description="最终生成图的名称。"
:required="false"
:default="null"
/>
<Params
name="context_schema"
type="dict"
description="最终生成图的 Context Schema。"
:required="false"
:default="null"
/>
<Params
name="input_schema"
type="dict"
description="最终生成图的输入 Schema。"
:required="false"
:default="null"
/>

<Params
name="output_schema"
type="dict"
description="最终生成图的输出 Schema。"
:required="false"
:default="null"
/>

<Params
name="checkpoint"
type="BaseCheckpointSaver"
description="LangGraph 的持久化 Checkpoint。"
:required="false"
:default="null"
/>

<Params
name="store"
type="BaseStore"
description="LangGraph 的持久化 Store。"
:required="false"
:default="null"
/>

<Params
name="cache"
type="BaseCache"
description="LangGraph 的 Cache。"
:required="false"
:default="null"
/>

**使用示例**：

开发一个软件项目，通常遵循一个严格的线性流程：
1. 需求分析：首先，产品经理必须明确“要做什么”，并产出详细的产品需求文档（PRD）。
2. 架构设计：然后，架构师基于PRD，设计“要怎么做”，规划系统蓝图和技术选型。
3. 代码编写：接着，开发工程师根据架构设计，将蓝图实现为具体的代码。
4. 测试与质保：最后，测试工程师对代码进行验证，确保其质量符合要求。
这个流程环环相扣，顺序不可颠倒。

对于上述四个流程，每个流程都有一个专门化的智能体负责。
1. 产品经理智能体：接收用户的模糊需求，输出结构化的产品需求文档（PRD）。
2. 架构师智能体：接收PRD，输出系统架构图和技术方案。
3. 开发工程师智能体：接收架构方案，输出可执行的源代码。
4. 测试工程师智能体：接收源代码，输出测试报告和优化建议。

通过 `create_sequential_pipeline` 函数，将这四个智能体无缝地串联起来，形成一个高度自动化、职责分明的软件开发流水线。


```python
from langchain.agents import AgentState
from langchain_core.messages import HumanMessage
from langchain_dev_utils.agents import create_agent
from langchain_dev_utils.pipeline import create_sequential_pipeline
from langchain_core.tools import tool
from langchain_dev_utils.chat_models import register_model_provider

register_model_provider(
    provider_name="vllm",
    chat_model="openai-compatible",
    base_url="http://localhost:8000/v1",
)

@tool
def analyze_requirements(user_request: str) -> str:
    """分析用户需求并生成详细的产品需求文档"""
    return f"根据用户请求'{user_request}'，已生成详细的产品需求文档，包含功能列表、用户故事和验收标准。"

@tool
def design_architecture(requirements: str) -> str:
    """根据需求文档设计系统架构"""
    return f"基于需求文档，已设计系统架构，包含微服务划分、数据流图和技术栈选择。"

@tool
def generate_code(architecture: str) -> str:
    """根据架构设计生成核心代码"""
    return f"基于架构设计，已生成核心业务代码，包含API接口、数据模型和业务逻辑实现。"

@tool
def create_tests(code: str) -> str:
    """为生成的代码创建测试用例"""
    return f"为生成的代码创建了单元测试、集成测试和端到端测试用例。"

# 构建自动化软件开发顺序工作流（管道）
graph = create_sequential_pipeline(
    sub_graphs=[
        create_agent(
            model="vllm:qwen3-4b",
            tools=[analyze_requirements],
            system_prompt="你是一个产品经理，负责分析用户需求并生成详细的产品需求文档。",
            name="requirements_agent",
        ),
        create_agent(
            model="vllm:qwen3-4b",
            tools=[design_architecture],
            system_prompt="你是一个系统架构师，负责根据需求文档设计系统架构。",
            name="architecture_agent",
        ),
        create_agent(
            model="vllm:qwen3-4b",
            tools=[generate_code],
            system_prompt="你是一个高级开发工程师，负责根据架构设计生成核心代码。",
            name="coding_agent",
        ),
        create_agent(
            model="vllm:qwen3-4b",
            tools=[create_tests],
            system_prompt="你是一个测试工程师，负责为生成的代码创建全面的测试用例。",
            name="testing_agent",
        ),
    ],
    state_schema=AgentState,
)

response = graph.invoke({"messages": [HumanMessage("开发一个电商网站，包含用户注册、商品浏览和购物车功能")]})
print(response)
```

最终生成的图结构如下：
![串行管道示意图](/img/sequential.png)

::: tip 📝
对于串行组合的图，langgraph 的 StateGraph 提供了 add_sequence 方法作为简便写法。该方法最适合在节点为函数（而非子图）时使用。若节点为子图，代码可能如下：

```python
graph = StateGraph(AgentState)
graph.add_sequence([("graph1", graph1), ("graph2", graph2), ("graph3", graph3)])
graph.add_edge("__start__", "graph1")
graph = graph.compile()
```

不过，上述写法仍显繁琐。因此，更推荐使用 `create_sequential_pipeline` 函数，它能通过一行代码快速构建串行执行图，更为简洁高效。
:::

## 并行编排

即用于搭建智能体并行工作流（Parallel Pipeline）。它的工作原理是将多个状态图并行组合，对于每个状态图并发地执行任务，从而提高任务的执行效率。

通过下面函数实现:

- `create_parallel_pipeline` - 以并行方式组合多个状态图

其参数如下:

<Params
name="sub_graphs"
type="list[StateGraph | CompiledStateGraph]"
description="要组合的状态图列表（必须是 StateGraph 或者 CompiledStateGraph 实例）。"
:required="true"
:default="null"
/>
<Params
name="state_schema"
type="dict"
description="最终生成图的 State Schema。"
:required="true"
:default="null"
/>
<Params
name="branches_fn"
type="Callable[[Any], list[Send]]"
description="并行分支函数，接收状态作为输入，返回 Send 对象列表以控制并行执行哪些子图。"
:required="false"
:default="null"
/>
<Params
name="graph_name"
type="string"
description="最终生成图的名称。"
:required="false"
:default="null"
/>
<Params
name="context_schema"
type="dict"
description="最终生成图的 Context Schema。"
:required="false"
:default="null"
/>
<Params
name="input_schema"
type="dict"
description="最终生成图的输入 Schema。"
:required="false"
:default="null"
/>

<Params
name="output_schema"
type="dict"
description="最终生成图的输出 Schema。"
:required="false"
:default="null"
/>

<Params
name="checkpoint"
type="BaseCheckpointSaver"
description="LangGraph 的持久化 Checkpoint。"
:required="false"
:default="null"
/>

<Params
name="store"
type="BaseStore"
description="LangGraph 的持久化 Store。"
:required="false"
:default="null"
/>

<Params
name="cache"
type="BaseCache"
description="LangGraph 的 Cache。"
:required="false"
:default="null"
/>

**使用示例**：

在软件开发中，当系统架构设计完成后，不同的功能模块往往可以由不同的团队或工程师同时进行开发，因为它们之间是相对独立的。这就是并行工作的典型场景。

假设要开发一个电商网站，其核心功能可以分为三个独立模块：
1. 用户模块（注册、登录、个人中心）
2. 商品模块（展示、搜索、分类）
3. 订单模块（下单、支付、状态查询）

如果串行开发，耗时将是三者之和。但如果并行开发，总耗时将约等于耗时最长的那一个模块的开发时间，效率大大提升。

通过`create_parallel_pipeline`函数，为每个模块分配一个专门的模块开发智能体，让它们并行工作。

```python
from langchain_dev_utils.pipeline import create_parallel_pipeline

@tool
def develop_user_module():
    """开发用户模块功能"""
    return "用户模块开发完成，包含注册、登录和个人资料管理功能。"

@tool
def develop_product_module():
    """开发商品模块功能"""
    return "商品模块开发完成，包含商品展示、搜索和分类功能。"

@tool
def develop_order_module():
    """开发订单模块功能"""
    return "订单模块开发完成，包含下单、支付和订单查询功能。"

# 构建前端模块开发的并行工作流（管道）
graph = create_parallel_pipeline(
    sub_graphs=[
        create_agent(
            model="vllm:qwen3-4b",
            tools=[develop_user_module],
            system_prompt="你是一个前端开发工程师，负责开发用户相关模块。",
            name="user_module_agent",
        ),
        create_agent(
            model="vllm:qwen3-4b",
            tools=[develop_product_module],
            system_prompt="你是一个前端开发工程师，负责开发商品相关模块。",
            name="product_module_agent",
        ),
        create_agent(
            model="vllm:qwen3-4b",
            tools=[develop_order_module],
            system_prompt="你是一个前端开发工程师，负责开发订单相关模块。",
            name="order_module_agent",
        ),
    ],
    state_schema=AgentState,
)
response = graph.invoke({"messages": [HumanMessage("并行开发电商网站的三个核心模块")]})
print(response)
```

最终生成的图结构如下：
![并行管道示意图](/img/parallel.png)


有些时候需要根据条件指定并行执行哪些子图，这时可以使用分支函数。
分支函数需要返回`Send`列表。

例如上述例子，假设开发的模块由用户指定，则只有被指定的模块才会被并行执行。

```python
# 构建并行管道（根据条件选择并行执行的子图）
from langgraph.types import Send

class DevAgentState(AgentState):
    """开发代理状态"""
    selected_modules: list[tuple[str, str]]


# 指定用户选择的模块
select_modules = [("user_module", "开发用户模块"), ("product_module", "开发商品模块")]

graph = create_parallel_pipeline(
    sub_graphs=[
        create_agent(
            model="vllm:qwen3-4b",
            tools=[develop_user_module],
            system_prompt="你是一个前端开发工程师，负责开发用户相关模块。",
            name="user_module_agent",
        ),
        create_agent(
            model="vllm:qwen3-4b",
            tools=[develop_product_module],
            system_prompt="你是一个前端开发工程师，负责开发商品相关模块。",
            name="product_module_agent",
        ),
        create_agent(
            model="vllm:qwen3-4b",
            tools=[develop_order_module],
            system_prompt="你是一个前端开发工程师，负责开发订单相关模块。",
            name="order_module_agent",
        ),
    ],
    state_schema=DevAgentState,
    branches_fn=lambda state: [
        Send(module_name + "_agent", {"messages": [HumanMessage(task_name)]})
        for module_name, task_name in state["selected_modules"]
    ],
)

response = graph.invoke(
    {
        "messages": [HumanMessage("开发电商网站的部分模块")],
        "selected_modules": select_modules,
    }
)
print(response)
```

**重要注意事项**

- 不传入 `branches_fn` 参数时，所有子图都会并行执行。
- 传入 `branches_fn` 参数时，执行哪些子图由该函数的返回值决定。