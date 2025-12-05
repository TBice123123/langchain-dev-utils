# 🦜️🧰 langchain-dev-utils

<p align="center">
    <em>A utility library for LangChain and LangGraph development.</em>
</p>

<p align="center">
  📚 <a href="https://tbice123123.github.io/langchain-dev-utils/">English</a> • 
  <a href="https://tbice123123.github.io/langchain-dev-utils/zh/">中文</a>
</p>

[![PyPI](https://img.shields.io/pypi/v/langchain-dev-utils.svg?color=%2334D058&label=pypi%20package)](https://pypi.org/project/langchain-dev-utils/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.11|3.12|3.13|3.14-%2334D058)](https://www.python.org/downloads)
[![Downloads](https://static.pepy.tech/badge/langchain-dev-utils/month)](https://pepy.tech/project/langchain-dev-utils)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://tbice123123.github.io/langchain-dev-utils/)

> This is the English version. For the Chinese version, please visit [中文文档](https://github.com/TBice123123/langchain-dev-utils/blob/master/README_cn.md)

**langchain-dev-utils** is a utility library focused on enhancing the development experience of LangChain and LangGraph. It provides a series of ready-to-use utility functions that can reduce repetitive code writing and improve code consistency and readability. By simplifying the development workflow, this library can help you build prototypes faster, iterate more smoothly, and create clearer and more reliable AI applications based on large language models.

## 🚀 Installation

```bash
pip install -U langchain-dev-utils

# Install full-featured version:
pip install -U langchain-dev-utils[standard]
```

## 📦 Core Features

### 1. **Model Management**

In `langchain`, the `init_chat_model`/`init_embeddings` functions can be used to initialize chat model instances/embedding model instances, but they support a limited number of model providers. This module provides registration functions (`register_model_provider`/`register_embeddings_provider`) to easily register any model provider for later use with `load_chat_model` / `load_embeddings` for model loading.

#### 1.1 Chat Model Management

There are two main functions:

- `register_model_provider`: Register a chat model provider
- `load_chat_model`: Load a chat model

Assuming you want to use the qwen3-4b model deployed with `vllm`, the reference code is as follows:

```python
from langchain_dev_utils.chat_models import (
    register_model_provider,
    load_chat_model,
)

# Register model provider
register_model_provider(
    provider_name="vllm",
    chat_model="openai-compatible",
    base_url="http://localhost:8000/v1",
)

# Load model
model = load_chat_model("vllm:qwen3-4b")
print(model.invoke("Hello"))
```

#### 1.2 Embedding Model Management

There are two main functions:

- `register_embeddings_provider`: Register an embedding model provider
- `load_embeddings`: Load an embedding model

Assuming you want to use the qwen3-embedding-4b model deployed with `vllm`, the reference code is as follows:

```python
from langchain_dev_utils.embeddings import register_embeddings_provider, load_embeddings

# Register embedding model provider
register_embeddings_provider(
    provider_name="vllm",
    embeddings_model="openai-compatible",
    base_url="http://localhost:8000/v1",
)

# Load embedding model
embeddings = load_embeddings("vllm:qwen3-embedding-4b")
emb = embeddings.embed_query("Hello")
print(emb)
```


### 2. **Message Conversion**

Includes the following features:

- Merge chain-of-thought content into final responses
- Stream content merging
- Content formatting tools

#### 2.1 Stream Content Merging

For streaming responses obtained using `stream()` and `astream()`, you can use `merge_ai_message_chunk` to merge them into a final AIMessage.

```python
from langchain_dev_utils.message_convert import merge_ai_message_chunk
chunks = list(model.stream("Hello"))
merged = merge_ai_message_chunk(chunks)
```

#### 2.2 Format List Content

For a list, you can use `format_sequence` to format it.

```python
from langchain_dev_utils.message_convert import format_sequence
text = format_sequence([
    "str1",
    "str2",
    "str3"
], separator="\n", with_num=True)
```


### 3. **Tool Calling**

Includes the following features:

- Check and parse tool calls
- Add human-in-the-loop functionality

#### 3.1 Check and Parse Tool Calls

`has_tool_calling` and `parse_tool_calling` are used to check and parse tool calls.

```python
import datetime
from langchain_core.tools import tool
from langchain_dev_utils.tool_calling import has_tool_calling, parse_tool_calling

@tool
def get_current_time() -> str:
    """Get current timestamp"""
    return str(datetime.datetime.now().timestamp())

response = model.bind_tools([get_current_time]).invoke("What time is it now?")

if has_tool_calling(response):
    name, args = parse_tool_calling(
        response, first_tool_call_only=True
    )
    print(name, args)
```

#### 3.2 Add Human-in-the-Loop Functionality

- `human_in_the_loop`: For synchronous tool functions
- `human_in_the_loop_async`: For asynchronous tool functions

Both can accept a `handler` parameter for custom breakpoint return and response handling logic.

```python
from langchain_dev_utils.tool_calling import human_in_the_loop
from langchain_core.tools import tool
import datetime

@human_in_the_loop
@tool
def get_current_time() -> str:
    """Get current timestamp"""
    return str(datetime.datetime.now().timestamp())
```


### 4. **Agent Development**

Includes the following features:

- Multi-agent construction
- Common middleware components

#### 4.1 Multi-Agent Construction

Wrapping agents as tools is a common implementation pattern in multi-agent systems, which is detailed in the official LangChain documentation. To this end, this library provides a pre-built function `wrap_agent_as_tool` to implement this pattern, which can wrap an agent instance into a tool that can be called by other agents.

Usage example:

```python
import datetime
from langchain_dev_utils.agents import create_agent, wrap_agent_as_tool
from langchain.agents import AgentState

@tool
def get_current_time() -> str:
    """Get current time"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

time_agent = create_agent("vllm:qwen3-4b", tools=[get_current_time], name="time-agent")
call_time_agent_tool = wrap_agent_as_tool(time_agent)  

agent = create_agent(
    "vllm:qwen3-4b",
    name="agent",
    tools=[call_time_agent_tool],
)
response = agent.invoke(
    {"messages": [{"role": "user", "content": "What time is it now?"}]}
)
print(response)
```

#### 4.2 Middleware

Provides some common middleware components. Below are examples using `ToolCallRepairMiddleware` and `PlanMiddleware`.

`ToolCallRepairMiddleware` is used to fix `invaild_tool_calls` content from large models.

`PlanMiddleware` is used for agent planning.

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
response = agent.invoke({"messages": [{"role": "user", "content": "Give me a travel plan to New York"}]}))
print(response)
```


### 5. **State Graph Orchestration**

Includes the following features:

- Sequential graph orchestration
- Parallel graph orchestration

#### 5.1 Sequential Graph Orchestration

Using `create_sequential_pipeline`, you can orchestrate multiple subgraphs in sequence:

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

# Build sequential pipeline (all subgraphs execute in sequence)
graph = create_sequential_pipeline(
    sub_graphs=[
        create_agent(
            model="vllm:qwen3-4b",
            tools=[get_current_time],
            system_prompt="You are a time query assistant, you can only answer the current time. If this question is not related to time, please directly answer that you cannot answer",
            name="time_agent",
        ),
        create_agent(
            model="vllm:qwen3-4b",
            tools=[get_current_weather],
            system_prompt="You are a weather query assistant, you can only answer the current weather. If this question is not related to weather, please directly answer that you cannot answer",
            name="weather_agent",
        ),
        create_agent(
            model="vllm:qwen3-4b",
            tools=[get_current_user],
            system_prompt="You are a user query assistant, you can only answer the current user. If this question is not related to users, please directly answer that you cannot answer",
            name="user_agent",
        ),
    ],
    state_schema=AgentState,
)

response = graph.invoke({"messages": [HumanMessage("Hello")]})
print(response)
```

#### 5.2 Parallel Graph Orchestration

Using `create_parallel_pipeline`, you can orchestrate multiple subgraphs in parallel:

```python
from langchain_dev_utils.pipeline import create_parallel_pipeline

# Build parallel pipeline (all subgraphs execute in parallel)
graph = create_parallel_pipeline(
    sub_graphs=[
        create_agent(
            model="vllm:qwen3-4b",
            tools=[get_current_time],
            system_prompt="You are a time query assistant, you can only answer the current time. If this question is not related to time, please directly answer that you cannot answer",
            name="time_agent",
        ),
        create_agent(
            model="vllm:qwen3-4b",
            tools=[get_current_weather],
            system_prompt="You are a weather query assistant, you can only answer the current weather. If this question is not related to weather, please directly answer that you cannot answer",
            name="weather_agent",
        ),
        create_agent(
            model="vllm:qwen3-4b",
            tools=[get_current_user],
            system_prompt="You are a user query assistant, you can only answer the current user. If this question is not related to users, please directly answer that you cannot answer",
            name="user_agent",
        ),
    ],
    state_schema=AgentState,
)
response = graph.invoke({"messages": [HumanMessage("Hello")]})
print(response)
```


## 💬 Join the Community

- [GitHub Repository](https://github.com/TBice123123/langchain-dev-utils) — Browse source code, submit Pull Requests
- [Issue Tracker](https://github.com/TBice123123/langchain-dev-utils/issues) — Report bugs or suggest improvements
- We welcome all forms of contributions — whether code, documentation, or usage examples. Let's build a more powerful and practical LangChain development ecosystem together