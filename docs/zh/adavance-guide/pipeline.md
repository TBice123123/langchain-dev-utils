# 状态图编排

## 概述

在LangGraph中，状态图编排是构建复杂 AI 应用的关键技术。通过将多个状态图按照特定模式进行组合，可以构建出功能强大、逻辑清晰的工作流。

本库提供了以下两种编排方式：

| 编排方式 | 功能描述 | 适用场景 |
|----------|----------|----------|
| **顺序编排** | 将多个状态图按照顺序方式进行编排，形成顺序工作流 | 任务需要按步骤依次执行，每个步骤依赖前一步骤的输出 |
| **并行编排** | 将多个状态图按照并行方式进行编排，形成并行工作流 | 多个任务相互独立，可以同时执行以提高效率 |

## 顺序编排

顺序编排（Sequential Pipeline）是一种将复杂任务分解为一系列连续、有序的子任务，并交由不同的专门化智能体依次处理的工作模式。

通过 `create_sequential_pipeline` 可将多个状态图以顺序编排方式进行组合。

### 典型应用场景

以软件开发流程为例，通常遵循一个严格的线性流程：

| 阶段 | 负责角色 | 输入 | 输出 |
|------|----------|------|------|
| 1. 需求分析 | 产品经理 | 用户需求 | 产品需求文档（PRD） |
| 2. 架构设计 | 架构师 | PRD | 系统架构图和技术方案 |
| 3. 代码编写 | 开发工程师 | 架构方案 | 可执行的源代码 |
| 4. 测试与质保 | 测试工程师 | 源代码 | 测试报告和优化建议 |

这个流程环环相扣，顺序不可颠倒。通过 `create_sequential_pipeline` 函数，可以将这四个智能体无缝地串联起来，形成一个高度自动化、职责分明的软件开发流水线。


### 基础示例

以下代码展示了如何使用 `create_sequential_pipeline` 构建软件开发流水线：

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
    return "基于需求文档，已设计系统架构，包含微服务划分、数据流图和技术栈选择。"


@tool
def generate_code(architecture: str) -> str:
    """根据架构设计生成核心代码"""
    return "基于架构设计，已生成核心业务代码，包含API接口、数据模型和业务逻辑实现。"


@tool
def create_tests(code: str) -> str:
    """为生成的代码创建测试用例"""
    return "为生成的代码创建了单元测试、集成测试和端到端测试用例。"


# 产品经理智能体
requirements_agent = create_agent(
    model="vllm:qwen3-4b",
    tools=[analyze_requirements],
    system_prompt="你是一个产品经理，负责分析用户需求并生成详细的产品需求文档。",
    name="requirements_agent",
)

# 系统架构师智能体
architecture_agent = create_agent(
    model="vllm:qwen3-4b",
    tools=[design_architecture],
    system_prompt="你是一个系统架构师，负责根据需求文档设计系统架构。",
    name="architecture_agent",
)

# 高级开发工程师智能体
coding_agent = create_agent(
    model="vllm:qwen3-4b",
    tools=[generate_code],
    system_prompt="你是一个高级开发工程师，负责根据架构设计生成核心代码。",
    name="coding_agent",
)

# 测试工程师智能体
testing_agent = create_agent(
    model="vllm:qwen3-4b",
    tools=[create_tests],
    system_prompt="你是一个测试工程师，负责为生成的代码创建全面的测试用例。",
    name="testing_agent",
)

# 构建自动化软件开发顺序工作流（管道）
graph = create_sequential_pipeline(
    sub_graphs=[
        requirements_agent,
        architecture_agent,
        coding_agent,
        testing_agent,
    ],
    state_schema=AgentState,
)

response = graph.invoke(
    {"messages": [HumanMessage("开发一个电商网站，包含用户注册、商品浏览和购物车功能")]}
)
print(response)
```

### 执行流程图

最终生成的图如下：

![Sequential Pipeline](../../assets/sequential.png)

### 上下文工程优化

上述基础示例仅作参考。实际上，该例子在运行时会将前面所有智能体的完整上下文依次传递给当前智能体，可能导致上下文膨胀，影响性能与效果。

推荐采用以下任一方案精简上下文：

| 方案 | 描述 | 优点 |
|------|------|------|
| **使用中间件** | 使用 `create_agent` 配合中间件，仅提取并传递必要信息 | 实现简单，代码改动小 |
| **自定义状态图** | 基于 `LangGraph` 完全自定义状态图，显式控制状态字段与消息流动 | 灵活性高，可精确控制 |

??? example "点击查看利用中间件解决的参考代码"

    ```python
    from typing import Any

    from langchain.agents import AgentState
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.messages import HumanMessage, RemoveMessage
    from langgraph.runtime import Runtime

    from langchain_dev_utils.agents import create_agent
    from langchain_dev_utils.agents.middleware import format_prompt
    from langchain_dev_utils.pipeline import create_sequential_pipeline


    class DeveloperState(AgentState, total=False):
        requirement: str
        architecture: str
        code: str
        tests: str

    class ClearAgentContextMiddleware(AgentMiddleware):
        state_schema = DeveloperState

        def __init__(self, result_save_key: str) -> None:
            super().__init__()
            self.result_save_key = result_save_key

        def after_agent(
            self, state: DeveloperState, runtime: Runtime
        ) -> dict[str, Any] | None:
            final_message = state["messages"][-1]
            update_key = self.result_save_key
            return {
                "messages": [
                    RemoveMessage(id=msg.id or "") for msg in state["messages"][1:]
                ],
                update_key: final_message.content,
            }

    # 产品经理智能体
    requirements_agent = create_agent(
        model="vllm:qwen3-4b",
        tools=[analyze_requirements],
        system_prompt="你是一个产品经理，负责分析用户需求并生成详细的产品需求文档。",
        name="requirements_agent",
        state_schema=DeveloperState,
        middleware=[format_prompt, ClearAgentContextMiddleware("requirement")],
    )

    # 系统架构师智能体
    architecture_agent = create_agent(
        model="vllm:qwen3-4b",
        tools=[design_architecture],
        system_prompt="你是一个系统架构师，负责根据需求文档设计系统架构。",
        name="architecture_agent",
        state_schema=DeveloperState,
        middleware=[format_prompt, ClearAgentContextMiddleware("architecture")],
    )

    # 高级开发工程师智能体
    coding_agent = create_agent(
        model="vllm:qwen3-4b",
        tools=[generate_code],
        system_prompt="你是一个高级开发工程师，负责根据架构设计生成核心代码。",
        name="coding_agent",
        state_schema=DeveloperState,
        middleware=[format_prompt, ClearAgentContextMiddleware("code")],
    )

    # 测试工程师智能体
    testing_agent = create_agent(
        model="vllm:qwen3-4b",
        tools=[create_tests],
        system_prompt="你是一个测试工程师，负责为生成的代码创建全面的测试用例。",
        name="testing_agent",
        state_schema=DeveloperState,
        middleware=[format_prompt, ClearAgentContextMiddleware("tests")],
    )

    # 构建自动化软件开发顺序工作流（管道）
    graph = create_sequential_pipeline(
        sub_graphs=[
            requirements_agent,
            architecture_agent,
            coding_agent,
            testing_agent,
        ],
        state_schema=DeveloperState,
    )

    response = graph.invoke(
        {"messages": [HumanMessage("开发一个电商网站，包含用户注册、商品浏览和购物车功能")]}
    )
    print(response)
    ```

    **实现说明**：

    1. **扩展状态模式**：在智能体的 State Schema 中添加了 `requirement`、`architecture`、`code`、`tests` 四个字段，用于存储对应智能体的最终输出结果。

    2. **自定义中间件**：创建了 `ClearAgentContextMiddleware` 中间件，在每个智能体结束后：
       - 清除当前的运行上下文（使用 `RemoveMessage`）
       - 将最终结果（`final_message.content`）保存到对应的字段中

    3. **动态提示格式化**：使用本库内置的 `format_prompt` 中间件，在运行时将前置智能体的输出按需动态拼入 `system_prompt`



!!! info "提示"

    对于串行组合的图，LangGraph 的 `StateGraph` 提供了 `add_sequence` 方法作为简便写法。该方法最适合在节点为函数（而非子图）时使用。

    ```python
    graph = StateGraph(AgentState)
    graph.add_sequence([("graph1", graph1), ("graph2", graph2), ("graph3", graph3)])
    graph.add_edge("__start__", "graph1")
    graph = graph.compile()
    ```

    不过，上述写法仍显繁琐。因此，更推荐使用 `create_sequential_pipeline` 函数，它能通过一行代码快速构建串行执行图，更为简洁高效。

## 并行编排

并行编排（Parallel Pipeline）通过将多个状态图并行组合，对每个状态图并发地执行任务，从而提高任务的执行效率。

通过 `create_parallel_pipeline` 函数，可将多个状态图以并行编排方式进行组合，实现并行执行任务的效果。

### 典型应用场景

在软件开发中，当系统架构设计完成后，不同的功能模块往往可以由不同的团队或工程师同时进行开发，因为它们之间是相对独立的。这就是并行工作的典型场景。

假设要开发一个电商网站，其核心功能可以分为三个独立模块：

| 模块 | 功能 | 开发内容 |
|------|------|----------|
| 用户模块 | 用户管理 | 注册、登录、个人中心 |
| 商品模块 | 商品管理 | 展示、搜索、分类 |
| 订单模块 | 订单管理 | 下单、支付、状态查询 |

如果串行开发，耗时将是三者之和。但如果并行开发，总耗时将约等于耗时最长的那一个模块的开发时间，效率大大提升。

### 基础示例

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


# 用户模块开发智能体
user_module_agent = create_agent(
    model="vllm:qwen3-4b",
    tools=[develop_user_module],
    system_prompt="你是一个前端开发工程师，负责开发用户相关模块。",
    name="user_module_agent",
)

# 商品模块开发智能体
product_module_agent = create_agent(
    model="vllm:qwen3-4b",
    tools=[develop_product_module],
    system_prompt="你是一个前端开发工程师，负责开发商品相关模块。",
    name="product_module_agent",
)

# 订单模块开发智能体
order_module_agent = create_agent(
    model="vllm:qwen3-4b",
    tools=[develop_order_module],
    system_prompt="你是一个前端开发工程师，负责开发订单相关模块。",
    name="order_module_agent",
)

# 构建前端模块开发的并行工作流（管道）
graph = create_parallel_pipeline(
    sub_graphs=[
        user_module_agent,
        product_module_agent,
        order_module_agent,
    ],
    state_schema=AgentState,
)
response = graph.invoke({"messages": [HumanMessage("并行开发电商网站的三个核心模块")]})
print(response)
```

### 执行流程图

最终生成的图如下：

![Parallel Pipeline](../../assets/parallel.png)

### 使用分支函数指定并行执行的子图

有些时候需要根据条件指定并行执行哪些子图，这时可以使用分支函数。分支函数需要返回 `Send` 列表。

#### 应用场景

例如上述例子，假设开发的模块由用户指定，则只有被指定的模块才会被并行执行。

```python
# 构建并行管道（根据条件选择并行执行的子图）
from langgraph.types import Send


class DevAgentState(AgentState):
    """开发代理状态"""

    selected_modules: list[tuple[str, str]]


# 指定用户选择的模块
select_modules = [("user_module", "开发用户模块"), ("product_module", "开发商品模块")]

user_module_agent = create_agent(
    model="vllm:qwen3-4b",
    tools=[develop_user_module],
    system_prompt="你是一个前端开发工程师，负责开发用户相关模块。",
    name="user_module_agent",
)

product_module_agent = create_agent(
    model="vllm:qwen3-4b",
    tools=[develop_product_module],
    system_prompt="你是一个前端开发工程师，负责开发商品相关模块。",
    name="product_module_agent",
)


order_module_agent = create_agent(
    model="vllm:qwen3-4b",
    tools=[develop_order_module],
    system_prompt="你是一个前端开发工程师，负责开发订单相关模块。",
    name="order_module_agent",
)


graph = create_parallel_pipeline(
    sub_graphs=[
        user_module_agent,
        product_module_agent,
        order_module_agent,
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

!!! tip "提示"

    - **不传入 `branches_fn` 参数时**：所有子图都会并行执行
    - **传入 `branches_fn` 参数时**：执行哪些子图由该函数的返回值决定