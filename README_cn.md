# langchain-dev-utils

[![PyPI](https://img.shields.io/pypi/v/langchain-dev-utils.svg)](https://pypi.org/project/langchain-dev-utils/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/your-username/langchain-dev-utils/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)

> 当前为中文版，英文版请访问[English Documentation](https://github.com/TBice123123/langchain-dev-utils/blob/master/README.md)

**langchain-dev-utils** 是一个专注于提升 LangChain 和 LangGraph 开发体验的实用工具库。它提供了一系列开箱即用的工具函数，既能减少重复代码编写，又能提高代码的一致性和可读性。通过简化开发工作流程，这个库可以帮助你更快地构建原型、更顺畅地进行迭代，并创建更清晰、更可靠的基于大语言模型的 AI 应用。

## 📚 文档

- [English Documentation](https://tbice123123.github.io/langchain-dev-utils-docs/en/)
- [中文文档](https://tbice123123.github.io/langchain-dev-utils-docs/zh/)

## 🚀 安装

```bash
pip install -U langchain-dev-utils

# 安装完整功能版：
pip install -U langchain-dev-utils[standard]
```

## 📦 核心功能

### 1. **模型管理**

在 `langchain` 中，`init_chat_model` 函数可用于初始化对话模型实例，但其支持的模型提供商较为有限。本模块提供了一个注册函数（`register_model_provider`/`register_embeddings_provider`），方便注册任意模型提供商，以便后续使用 `load_chat_model` / `load_embeddings` 进行模型加载。

`register_model_provider` 参数说明：

- `provider_name`：模型提供商名称，作为后续模型加载的标识
- `chat_model`：对话模型，可以是 ChatModel 或字符串（目前支持 "openai-compatible"）
- `base_url`：模型提供商的 API 地址

`register_embeddings_provider` 参数说明：

- `provider_name`：嵌入模型提供商名称，作为后续模型加载的标识
- `embeddings_model`：嵌入模型，可以是 Embeddings 或字符串（目前支持 "openai-compatible"）
- `base_url`：模型提供商的 API 地址

使用示例：

```python
# 对话模型管理
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

嵌入模型使用：

```python
from langchain_dev_utils.embeddings import register_embeddings_provider, load_embeddings

register_embeddings_provider(
    provider_name="vllm",
    embeddings_model="openai-compatible",
    base_url="http://localhost:8000/v1",
)
embeddings = load_embeddings("vllm:qwen3-embedding-4b")
emb = embeddings.embed_query("Hello")
print(emb)
```

**了解更多**：[模型管理](https://tbice123123.github.io/langchain-dev-utils-docs/zh/model-management.html)

### 2. **消息转换**

包含以下功能：

- 将思维链内容合并到最终响应中
- 流式内容合并
- 内容格式化工具

将思维链内容合并到最终响应：

```python
from langchain_dev_utils.message_convert import (
    convert_reasoning_content_for_ai_message,
    convert_reasoning_content_for_chunk_iterator,
    merge_ai_message_chunk,
    format_sequence
)

response = model.invoke("Hello")

cleaned = convert_reasoning_content_for_ai_message(
    response, think_tag=("<think>", "</think>")
)

for chunk in convert_reasoning_content_for_chunk_iterator(
    model.stream("Hello")
):
    print(chunk.content, end="", flush=True)
```

合并流式响应：

```python
chunks = list(model.stream("Hello"))
merged = merge_ai_message_chunk(chunks)
```

格式化序列：

```python
text = format_sequence([
    "str1",
    "str2",
    "str3"
], separator="\n", with_num=True)
```

**了解更多**：[消息转换](https://tbice123123.github.io/langchain-dev-utils-docs/zh/message-conversion.html)

### 3. **工具调用**

包含以下功能：

- 检查和解析工具调用
- 添加人机交互功能

使用示例：

```python
import datetime
from langchain_core.tools import tool
from langchain_dev_utils.tool_calling import has_tool_calling, parse_tool_calling, human_in_the_loop
from langchain_core.messages import AIMessage
from typing import cast

@human_in_the_loop
def get_current_time() -> str:
    """获取当前时间戳"""
    return str(datetime.datetime.now().timestamp())

response = model.bind_tools([get_current_time]).invoke("现在几点了？")

if has_tool_calling(cast(AIMessage, response)):
    name, args = parse_tool_calling(
        cast(AIMessage, response), first_tool_call_only=True
    )
    print(name, args)
```

**了解更多**：[工具调用](https://tbice123123.github.io/langchain-dev-utils-docs/zh/tool-calling.html)

### 4. **智能体开发**

包含以下功能：

- 预设的智能体工厂函数
- 常用的中间件组件

使用示例：

```python
from langchain_dev_utils.agents import create_agent
from langchain.agents import AgentState

agent = create_agent("vllm:qwen3-4b", tools=[get_current_time], name="time-agent")
response = agent.invoke({"messages": [{"role": "user", "content": "现在几点了？"}]})
print(response)
```

中间件使用：

```python
from langchain_dev_utils.agents.middleware import (
    SummarizationMiddleware,
    LLMToolSelectorMiddleware,
    PlanMiddleware,
)

agent=create_agent(
    "vllm:qwen3-4b",
    name="plan-agent",
    middleware=[PlanMiddleware(), SummarizationMiddleware(), LLMToolSelectorMiddleware()]
)
response = agent.invoke({"messages": [{"role": "user", "content": "给我一个去纽约旅行的计划"}]}))
print(response)
```

**了解更多**：[智能体开发](https://tbice123123.github.io/langchain-dev-utils-docs/zh/agent-development.html)

### 5. **状态图编排**

包含以下功能：

- 顺序图编排
- 并行图编排

```python
from langchain_dev_utils.pipeline import sequential_pipeline, parallel_pipeline

# 构建顺序流程
graph = sequential_pipeline(
    sub_graphs=[
        make_graph("graph1"),
        make_graph("graph2"),
        make_graph("graph3"),
    ],
    state_schema=State,
)

# 构建并行流程
graph = parallel_pipeline(
    sub_graphs=[
        make_graph("graph1"),
        make_graph("graph2"),
        make_graph("graph3"),
    ],
    state_schema=State,
)
```

**了解更多**：[状态图编排](https://tbice123123.github.io/langchain-dev-utils-docs/zh/graph-orchestration.html)

## 💬 加入社区

- 🐙 [GitHub 仓库](https://github.com/TBice123123/langchain-dev-utils) — 浏览源代码，提交 Pull Request
- 🐞 [问题追踪](https://github.com/TBice123123/langchain-dev-utils/issues) — 报告 Bug 或提出改进建议
- 💡 我们欢迎各种形式的贡献 —— 无论是代码、文档还是使用示例。让我们一起构建一个更强大、更实用的 LangChain 开发生态系统！
