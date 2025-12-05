# 🦜️🧰 langchain-dev-utils

<p align="center">
    <em>用于 LangChain 和 LangGraph 开发的实用工具库。</em>
</p>

<p align="center">
  📚 <a href="https://tbice123123.github.io/langchain-dev-utils/">English</a> • 
  <a href="https://tbice123123.github.io/langchain-dev-utils/zh/">中文</a>
</p>

[![PyPI](https://img.shields.io/pypi/v/langchain-dev-utils.svg?color=%2334D058&label=pypi%20package)](https://pypi.org/project/langchain-dev-utils/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.11|3.12|3.13|3.14-%2334D058)](https://www.python.org/downloads)
[![Downloads](https://static.pepy.tech/badge/langchain-dev-utils/month)](https://pepy.tech/project/langchain-dev-utils)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://tbice123123.github.io/langchain-dev-utils/zh/)

> 当前为中文版，英文版请访问[English Documentation](https://github.com/TBice123123/langchain-dev-utils/blob/master/README.md)

**langchain-dev-utils** 是一个专注于提升 LangChain 和 LangGraph 开发体验的实用工具库。它提供了一系列开箱即用的工具函数，既能减少重复代码编写，又能提高代码的一致性和可读性。通过简化开发工作流程，这个库可以帮助你更快地构建原型、更顺畅地进行迭代，并创建更清晰、更可靠的基于大语言模型的 AI 应用。

## 🚀 安装

```bash
pip install -U langchain-dev-utils

# 安装完整功能版：
pip install -U langchain-dev-utils[standard]
```

## 📦 核心功能

### 1. **模型管理**

在 `langchain` 中，`init_chat_model`/`init_embeddings` 函数可用于初始化对话模型实例/嵌入模型实例，但其支持的模型提供商较为有限。本模块提供了一个注册函数（`register_model_provider`/`register_embeddings_provider`），方便注册任意模型提供商，以便后续使用 `load_chat_model` / `load_embeddings` 进行模型加载。

#### 1.1 对话模型管理

主要有以下两个函数：

- `register_model_provider`：注册对话模型提供商
- `load_chat_model`：加载对话模型

假设接入使用`vllm`部署的 qwen3-4b 模型，则参考代码如下：

```python
from langchain_dev_utils.chat_models import (
    register_model_provider,
    load_chat_model,
)

# 注册模型提供商
register_model_provider(
    provider_name="vllm",
    chat_model="openai-compatible",
    base_url="http://localhost:8000/v1",
)

# 加载模型
model = load_chat_model("vllm:qwen3-4b")
print(model.invoke("Hello"))
```

#### 1.2 嵌入模型管理

主要有以下两个函数：

- `register_embeddings_provider`：注册嵌入模型提供商
- `load_embeddings`：加载嵌入模型

假设接入使用`vllm`部署的 qwen3-embedding-4b 模型，则参考代码如下：

```python
from langchain_dev_utils.embeddings import register_embeddings_provider, load_embeddings

# 注册嵌入模型提供商
register_embeddings_provider(
    provider_name="vllm",
    embeddings_model="openai-compatible",
    base_url="http://localhost:8000/v1",
)

# 加载嵌入模型
embeddings = load_embeddings("vllm:qwen3-embedding-4b")
emb = embeddings.embed_query("Hello")
print(emb)
```


### 2. **消息转换**

包含以下功能：

- 将思维链内容合并到最终响应中
- 流式内容合并
- 内容格式化工具

#### 2.1 流式内容合并

对于使用`stream()`和`astream()`所获得的流式响应，可以使用`merge_ai_message_chunk`进行合并为一个最终的 AIMessage。

```python
from langchain_dev_utils.message_convert import merge_ai_message_chunk
chunks = list(model.stream("Hello"))
merged = merge_ai_message_chunk(chunks)
```

#### 2.2 格式化列表内容

对于一个列表，可以使用`format_sequence`进行格式化。

```python
from langchain_dev_utils.message_convert import format_sequence
text = format_sequence([
    "str1",
    "str2",
    "str3"
], separator="\n", with_num=True)
```


### 3. **工具调用**

包含以下功能：

- 检查和解析工具调用
- 添加人机交互功能

#### 3.1 检查和解析工具调用

`has_tool_calling`和`parse_tool_calling`用于检查和解析工具调用。

```python
import datetime
from langchain_core.tools import tool
from langchain_dev_utils.tool_calling import has_tool_calling, parse_tool_calling

@tool
def get_current_time() -> str:
    """获取当前时间戳"""
    return str(datetime.datetime.now().timestamp())

response = model.bind_tools([get_current_time]).invoke("现在几点了？")

if has_tool_calling(response):
    name, args = parse_tool_calling(
        response, first_tool_call_only=True
    )
    print(name, args)
```

#### 3.2 添加人机交互功能

- `human_in_the_loop`：用于同步工具函数
- `human_in_the_loop_async`：用于异步工具函数

其中都可以传递`handler`参数，用于自定义断点返回和响应处理逻辑。

```python
from langchain_dev_utils.tool_calling import human_in_the_loop
from langchain_core.tools import tool
import datetime

@human_in_the_loop
@tool
def get_current_time() -> str:
    """获取当前时间戳"""
    return str(datetime.datetime.now().timestamp())
```


### 4. **智能体开发**

包含以下功能：

- 多智能体构建
- 常用的中间件组件

#### 4.1 多智能体构建

将智能体封装为工具是多智能体系统中的一种常见实现模式，LangChain 官方文档对此有详细阐述。为此，本库提供了预构建函数`wrap_agent_as_tool` 来实现此模式，该函数能够将一个智能体实例封装成一个可供其它智能体调用的工具。

使用示例：

```python
import datetime
from langchain_dev_utils.agents import create_agent, wrap_agent_as_tool
from langchain.agents import AgentState

@tool
def get_current_time() -> str:
    """获取当前时间"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

time_agent = create_agent("vllm:qwen3-4b", tools=[get_current_time], name="time-agent")
call_time_agent_tool = wrap_agent_as_tool(time_agent)  

agent = create_agent(
    "vllm:qwen3-4b",
    name="agent",
    tools=[call_time_agent_tool],
)
response = agent.invoke(
    {"messages": [{"role": "user", "content": "现在几点了？"}]}
)
print(response)
```

#### 4.2 中间件

提供了一些常用的中间件组件。下面以`ToolCallRepairMiddleware`和`PlanMiddleware`为例。

`ToolCallRepairMiddleware`用于大模型的 `invaild_tool_calls` 内容的修复。

`PlanMiddleware`用于智能体的计划。

```python
from langchain_dev_utils.agents.middleware import (
    ToolCallRepairMiddleware,
    PlanMiddleware,
)

agent=create_agent(
    "vllm:qwen3-4b",
    name="plan-agent",
    middleware=[ToolCallRepairMiddleware(), PlanMiddleware(
        use_read_plan_tool=False
    )]
)
response = agent.invoke({"messages": [{"role": "user", "content": "给我一个去纽约旅行的计划"}]}))
print(response)
```


### 5. **状态图编排**

包含以下功能：

- 顺序图编排
- 并行图编排

#### 5.1 顺序图编排

利用`create_sequential_pipeline`可以将多个子图按照顺序进行编排：

```python
from langchain.agents import AgentState
from langchain_core.messages import HumanMessage
from langchain_dev_utils.agents import create_agent
from langchain_dev_utils.pipeline import create_sequential_pipeline
from langchain_dev_utils.chat_models import register_model_provider

register_model_provider(
    provider_name="vllm",
    chat_model="openai-compatible",
    base_url="http://localhost:8000/v1",
)

# 构建顺序管道（所有子图顺序执行）
graph = create_sequential_pipeline(
    sub_graphs=[
        create_agent(
            model="vllm:qwen3-4b",
            tools=[get_current_time],
            system_prompt="你是一个时间查询助手,仅能回答当前时间,如果这个问题和时间无关,请直接回答我无法回答",
            name="time_agent",
        ),
        create_agent(
            model="vllm:qwen3-4b",
            tools=[get_current_weather],
            system_prompt="你是一个天气查询助手,仅能回答当前天气,如果这个问题和天气无关,请直接回答我无法回答",
            name="weather_agent",
        ),
        create_agent(
            model="vllm:qwen3-4b",
            tools=[get_current_user],
            system_prompt="你是一个用户查询助手,仅能回答当前用户,如果这个问题和用户无关,请直接回答我无法回答",
            name="user_agent",
        ),
    ],
    state_schema=AgentState,
)

response = graph.invoke({"messages": [HumanMessage("你好")]})
print(response)
```

#### 5.2 并行图编排

利用`create_parallel_pipeline`可以将多个子图按照并行进行编排：

```python
from langchain_dev_utils.pipeline import create_parallel_pipeline

# 构建并行管道（所有子图并行执行）
graph = create_parallel_pipeline(
    sub_graphs=[
        create_agent(
            model="vllm:qwen3-4b",
            tools=[get_current_time],
            system_prompt="你是一个时间查询助手,仅能回答当前时间,如果这个问题和时间无关,请直接回答我无法回答",
            name="time_agent",
        ),
        create_agent(
            model="vllm:qwen3-4b",
            tools=[get_current_weather],
            system_prompt="你是一个天气查询助手,仅能回答当前天气,如果这个问题和天气无关,请直接回答我无法回答",
            name="weather_agent",
        ),
        create_agent(
            model="vllm:qwen3-4b",
            tools=[get_current_user],
            system_prompt="你是一个用户查询助手,仅能回答当前用户,如果这个问题和用户无关,请直接回答我无法回答",
            name="user_agent",
        ),
    ],
    state_schema=AgentState,
)
response = graph.invoke({"messages": [HumanMessage("你好")]})
print(response)
```


## 💬 加入社区

- [GitHub 仓库](https://github.com/TBice123123/langchain-dev-utils) — 浏览源代码，提交 Pull Request
- [问题追踪](https://github.com/TBice123123/langchain-dev-utils/issues) — 报告 Bug 或提出改进建议
- 我们欢迎各种形式的贡献 —— 无论是代码、文档还是使用示例。让我们一起构建一个更强大、更实用的 LangChain 开发生态系统！
