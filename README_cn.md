# 🦜️🧰 langchain-dev-utils

<p align="center">
    <em>用于 LangChain 和 LangGraph 开发的实用工具库。</em>
</p>

<p align="center">
  📚 <a href="https://tbice123123.github.io/langchain-dev-utils-docs/en/">English</a> • 
  <a href="https://tbice123123.github.io/langchain-dev-utils-docs/zh/">中文</a>
</p>

[![PyPI](https://img.shields.io/pypi/v/langchain-dev-utils.svg?color=%2334D058&label=pypi%20package)](https://pypi.org/project/langchain-dev-utils/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.11|3.12|3.13|3.14-%2334D058)](https://www.python.org/downloads)
[![Downloads](https://static.pepy.tech/badge/langchain-dev-utils/month)](https://pepy.tech/project/langchain-dev-utils)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://tbice123123.github.io/langchain-dev-utils-docs/zh/)

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

`register_model_provider` 参数说明：

- `provider_name`：模型提供商名称，作为后续模型加载的标识
- `chat_model`：对话模型，可以是 ChatModel 或字符串（目前支持 "openai-compatible"）
- `base_url`：模型提供商的 API 地址（可选，当 `chat_model` 为字符串且是"openai-compatible"时有效）
- `provider_config`：模型提供商的相关配置（可选，当 `chat_model` 为字符串且是 "openai-compatible" 时有效），可以配置一些提供商的相关参数，例如是否支持 json_mode 的结构化输出方式、支持的 tool_choice 列表等

`load_chat_model` 参数说明：

- `model`：对话模型名称，类型为 str
- `model_provider`：对话模型提供商名称，类型为 str，可选
- `kwargs`：传递给对话模型类的额外的参数，例如 temperature、top_p 等

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

`register_embeddings_provider` 参数说明：

- `provider_name`：嵌入模型提供商名称，作为后续模型加载的标识
- `embeddings_model`：嵌入模型，可以是 Embeddings 或字符串（目前支持 "openai-compatible"）
- `base_url`：模型提供商的 API 地址（可选，当 `embeddings_model` 为字符串且是"openai-compatible"时有效）

`load_embeddings` 参数说明：

- `model`：嵌入模型名称，类型为 str
- `provider`：嵌入模型提供商名称，类型为 str，可选
- `kwargs`：其它额外的参数

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

**对于更多关于模型管理的相关介绍，请参考**: [对话模型管理](https://tbice123123.github.io/langchain-dev-utils-docs/zh/model-management/chat.html)、[嵌入模型管理](https://tbice123123.github.io/langchain-dev-utils-docs/zh/model-management/embedding.html)

### 2. **消息转换**

包含以下功能：

- 将思维链内容合并到最终响应中
- 流式内容合并
- 内容格式化工具

#### 2.1 流式内容合并

对于使用`stream()`和`astream()`所获得的流式响应，可以使用`merge_ai_message_chunk`进行合并为一个最终的 AIMessage。

`merge_ai_message_chunk` 参数说明：

- `chunks`：AIMessageChunk 列表

```python
chunks = list(model.stream("Hello"))
merged = merge_ai_message_chunk(chunks)
```

#### 2.2 格式化列表内容

对于一个列表，可以使用`format_sequence`进行格式化。

`format_sequence` 参数说明：

- `inputs`：包含以下任意类型的列表：
  - langchain_core.messages：HumanMessage、AIMessage、SystemMessage、ToolMessage
  - langchain_core.documents.Document
  - str
- `separator`：用于连接内容的字符串，默认为 "-"。
- `with_num`：如果为 True，为每个项目添加数字前缀（例如 "1. 你好"），默认为 False。

```python
text = format_sequence([
    "str1",
    "str2",
    "str3"
], separator="\n", with_num=True)
```

**对于更多关于消息转换的相关介绍，请参考**: [消息处理](https://tbice123123.github.io/langchain-dev-utils-docs/zh/message-conversion/message.html),[格式化列表内容](https://tbice123123.github.io/langchain-dev-utils-docs/zh/message-conversion/format.html)

### 3. **工具调用**

包含以下功能：

- 检查和解析工具调用
- 添加人机交互功能

#### 3.1 检查和解析工具调用

`has_tool_calling`和`parse_tool_calling`用于检查和解析工具调用。

`has_tool_calling` 参数说明：

- `message`：AIMessage 对象

`parse_tool_calling` 参数说明：

- `message`：AIMessage 对象
- `first_tool_call_only`：是否只检查第一个工具调用

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
from langchain_dev_utils import human_in_the_loop
from langchain_core.tools import tool
import datetime

@human_in_the_loop
@tool
def get_current_time() -> str:
    """获取当前时间戳"""
    return str(datetime.datetime.now().timestamp())
```

**对于更多关于工具调用的相关介绍，请参考**: [添加人在回路支持](https://tbice123123.github.io/langchain-dev-utils-docs/zh/tool-calling/human-in-the-loop.html),[工具调用处理](https://tbice123123.github.io/langchain-dev-utils-docs/zh/tool-calling/tool.html)

### 4. **智能体开发**

包含以下功能：

- 预设的智能体工厂函数
- 常用的中间件组件

#### 4.1 智能体工厂函数

LangChain v1 版本中，官方提供的 `create_agent` 函数可以用于创建单智能体，其中 model 参数支持传入 BaseChatModel 实例或特定字符串（当传入字符串时，仅限于 `init_chat_model` 支持的模型）。为扩展字符串指定模型的灵活性，本库提供了功能相同的 `create_agent` 函数，使您能直接使用 `load_chat_model` 支持的模型（需要提取注册）。

使用示例：

```python
from langchain_dev_utils.agents import create_agent
from langchain.agents import AgentState

agent = create_agent("vllm:qwen3-4b", tools=[get_current_time], name="time-agent")
response = agent.invoke({"messages": [{"role": "user", "content": "现在几点了？"}]})
print(response)
```

#### 4.2 中间件

提供了一些常用的中间件组件。下面以`SummarizationMiddleware`和`PlanMiddleware`为例。

`SummarizationMiddleware`用于智能体的总结。

`PlanMiddleware`用于智能体的计划。

```python
from langchain_dev_utils.agents.middleware import (
    SummarizationMiddleware,
    PlanMiddleware,
)

agent=create_agent(
    "vllm:qwen3-4b",
    name="plan-agent",
    middleware=[PlanMiddleware(), SummarizationMiddleware(model="vllm:qwen3-4b")]
)
response = agent.invoke({"messages": [{"role": "user", "content": "给我一个去纽约旅行的计划"}]}))
print(response)
```

**对于更多关于智能体开发以及所有的内置中间件的相关介绍，请参考**: [预构建智能体函数](https://tbice123123.github.io/langchain-dev-utils-docs/zh/agent-development/prebuilt.html),[中间件](https://tbice123123.github.io/langchain-dev-utils-docs/zh/agent-development/middleware.html)

### 5. **状态图编排**

包含以下功能：

- 顺序图编排
- 并行图编排

#### 5.1 顺序图编排

顺序图编排：
采用`sequential_pipeline`，支持的参数如下:

- `sub_graphs`: 要组合的状态图列表（必须是 StateGraph 实例）
- `state_schema`: 最终生成图的 State Schema
- `graph_name`: 最终生成图的名称（可选）
- `context_schema`: 最终生成图的 Context Schema（可选）
- `input_schema`: 最终生成图的输入 Schema（可选）
- `output_schema`: 最终生成图的输出 Schema（可选）
- `checkpoint`: LangGraph 的持久化 Checkpoint（可选）
- `store`: LangGraph 的持久化 Store（可选）
- `cache`: LangGraph 的 Cache（可选）

```python
from langchain.agents import AgentState
from langchain_core.messages import HumanMessage
from langchain_dev_utils.agents import create_agent
from langchain_dev_utils.pipeline import sequential_pipeline
from langchain_dev_utils.chat_models import register_model_provider

register_model_provider(
    provider_name="vllm",
    chat_model="openai-compatible",
    base_url="http://localhost:8000/v1",
)

# 构建顺序管道（所有子图顺序执行）
graph = sequential_pipeline(
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

并行图编排：
采用`parallel_pipeline`，支持的参数如下:

- `sub_graphs`: 要组合的状态图列表
- `state_schema`: 最终生成图的 State Schema
- `branches_fn`: 并行分支函数，返回 Send 对象列表控制并行执行
- `graph_name`: 最终生成图的名称（可选）
- `context_schema`: 最终生成图的 Context Schema（可选）
- `input_schema`: 最终生成图的输入 Schema（可选）
- `output_schema`: 最终生成图的输出 Schema（可选）
- `checkpoint`: LangGraph 的持久化 Checkpoint（可选）
- `store`: LangGraph 的持久化 Store（可选）
- `cache`: LangGraph 的 Cache（可选）

```python
from langchain_dev_utils.pipeline import parallel_pipeline

# 构建并行管道（所有子图并行执行）
graph = parallel_pipeline(
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

**对于更多关于状态图编排的相关介绍，请参考**: [状态图编排管道](https://tbice123123.github.io/langchain-dev-utils-docs/zh/graph-orchestration/pipeline.html)

## 💬 加入社区

- [GitHub 仓库](https://github.com/TBice123123/langchain-dev-utils) — 浏览源代码，提交 Pull Request
- [问题追踪](https://github.com/TBice123123/langchain-dev-utils/issues) — 报告 Bug 或提出改进建议
- 我们欢迎各种形式的贡献 —— 无论是代码、文档还是使用示例。让我们一起构建一个更强大、更实用的 LangChain 开发生态系统！
