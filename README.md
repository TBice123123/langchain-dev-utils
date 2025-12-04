# 🦜️🧰 langchain-dev-utils

<p align="center">
    <em>A utility library for LangChain and LangGraph development.</em>
</p>

<p align="center">
  📚 <a href="https://tbice123123.github.io/langchain-dev-utils-docs/en/">English</a> • 
  <a href="https://tbice123123.github.io/langchain-dev-utils-docs/zh/">中文</a>
</p>

[![PyPI](https://img.shields.io/pypi/v/langchain-dev-utils.svg?color=%2334D058&label=pypi%20package)](https://pypi.org/project/langchain-dev-utils/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.11|3.12|3.13|3.14-%2334D058)](https://www.python.org/downloads)
[![Downloads](https://static.pepy.tech/badge/langchain-dev-utils/month)](https://pepy.tech/project/langchain-dev-utils)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://tbice123123.github.io/langchain-dev-utils-docs/en/)

> This is the English version. For the Chinese version, please visit [Chinese Documentation](https://github.com/TBice123123/langchain-dev-utils/blob/master/README_cn.md)

**langchain-dev-utils** is a utility library focused on enhancing the development experience with LangChain and LangGraph. It provides a series of out-of-the-box utility functions that can both reduce repetitive code writing and improve code consistency and readability. By simplifying development workflows, this library helps you prototype faster, iterate more smoothly, and create clearer, more reliable LLM-based AI applications.

## 🚀 Installation

```bash
pip install -U langchain-dev-utils

# Install the full-featured version:
pip install -U langchain-dev-utils[standard]
```

## 📦 Core Features

### 1. **Model Management**

In `langchain`, the `init_chat_model`/`init_embeddings` functions can be used to initialize chat model instances/embedding model instances, but the model providers they support are relatively limited. This module provides a registration function (`register_model_provider`/`register_embeddings_provider`) to register any model provider for subsequent model loading using `load_chat_model` / `load_embeddings`.

#### 1.1 Chat Model Management

Mainly consists of the following two functions:

- `register_model_provider`: Register a chat model provider
- `load_chat_model`: Load a chat model

Example for integrating a qwen3-4b model deployed using `vllm`:

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

Mainly consists of the following two functions:

- `register_embeddings_provider`: Register an embedding model provider
- `load_embeddings`: Load an embedding model

Example for integrating a qwen3-embedding-4b model deployed using `vllm`:

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

**For more information about model management, please refer to**: [Chat Model Management](https://tbice123123.github.io/langchain-dev-utils-docs/en/model-management/chat.html), [Embedding Model Management](https://tbice123123.github.io/langchain-dev-utils-docs/en/model-management/embedding.html)

### 2. **Message Conversion**

Includes the following features:

- Merge reasoning content into the final response
- Stream content merging
- Content formatting tools

#### 2.1 Stream Content Merging

For stream responses obtained using `stream()` and `astream()`, you can use `merge_ai_message_chunk` to merge them into a final AIMessage.

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

**For more information about message conversion, please refer to**: [Message Process](https://tbice123123.github.io/langchain-dev-utils-docs/en/message-conversion/message.html), [Formatting List Content](https://tbice123123.github.io/langchain-dev-utils-docs/en/message-conversion/format.html)

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
    """Get the current timestamp"""
    return str(datetime.datetime.now().timestamp())

response = model.bind_tools([get_current_time]).invoke("What time is it?")

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
    """Get the current timestamp"""
    return str(datetime.datetime.now().timestamp())
```

**For more information about tool calling, please refer to**: [Add Human-in-the-Loop Support](https://tbice123123.github.io/langchain-dev-utils-docs/en/tool-calling/human-in-the-loop.html), [Tool Call Handling](https://tbice123123.github.io/langchain-dev-utils-docs/en/tool-calling/tool.html)

### 4. **Agent Development**

Includes the following capabilities:

- Multi-agent construction  
- Commonly used middleware components  

#### 4.1 Multi-Agent Construction

Wrapping an agent as a tool is a common implementation pattern in multi-agent systems, as elaborated in the official LangChain documentation. To support this pattern, this library provides a pre-built utility function `wrap_agent_as_tool`, which encapsulates an agent instance into a tool that can be invoked by other agents.

**Usage Example**:

```python
import datetime
from langchain_dev_utils.agents import create_agent, wrap_agent_as_tool
from langchain.agents import AgentState


@tool
def get_current_time() -> str:
    """Get the current time."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


agent = create_agent("vllm:qwen3-4b", tools=[get_current_time], name="time-agent")
call_time_agent_tool = wrap_agent_as_tool(agent)
response = call_time_agent_tool.invoke(
    {"messages": [{"role": "user", "content": "What time is it now?"}]}
)
print(response)
```

#### 4.2 Middleware

Provides several commonly used middleware components. Below are examples using `ToolCallRepairMiddleware` and `PlanMiddleware`.

- `ToolCallRepairMiddleware` automatically repairs malformed tool calls found in the model's `invalid_tool_calls` output.
- `PlanMiddleware` enables task planning capabilities for agents.

```python
from langchain_dev_utils.agents.middleware import (
    ToolCallRepairMiddleware,
    PlanMiddleware,
)

agent = create_agent(
    "vllm:qwen3-4b",
    name="plan-agent",
    middleware=[
        ToolCallRepairMiddleware(),
        PlanMiddleware(use_read_plan_tool=False)
    ]
)
response = agent.invoke({"messages": [{"role": "user", "content": "Give me a travel plan for visiting New York."}]})
print(response)
```

**For more details on agent development and a complete list of built-in middleware, please refer to**:  
[Multi-Agent Construction](https://tbice123123.github.io/langchain-dev-utils-docs/en/agent-development/multi-agent.html),  
[Middleware](https://tbice123123.github.io/langchain-dev-utils-docs/en/agent-development/middleware.html)


### 5. **State Graph Orchestration**

Includes the following capabilities:

- Sequential graph orchestration
- Parallel graph orchestration

#### 5.1 Sequential Graph Orchestration

Use `create_sequential_pipeline` to orchestrate multiple subgraphs in sequential order:

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

# Build a sequential pipeline (all subgraphs executed in order)
graph = create_sequential_pipeline(
    sub_graphs=[
        create_agent(
            model="vllm:qwen3-4b",
            tools=[get_current_time],
            system_prompt="You are a time-query assistant. You can only answer questions about the current time. If the question is unrelated to time, respond with 'I cannot answer that.'",
            name="time_agent",
        ),
        create_agent(
            model="vllm:qwen3-4b",
            tools=[get_current_weather],
            system_prompt="You are a weather-query assistant. You can only answer questions about the current weather. If the question is unrelated to weather, respond with 'I cannot answer that.'",
            name="weather_agent",
        ),
        create_agent(
            model="vllm:qwen3-4b",
            tools=[get_current_user],
            system_prompt="You are a user-query assistant. You can only answer questions about the current user. If the question is unrelated to the user, respond with 'I cannot answer that.'",
            name="user_agent",
        ),
    ],
    state_schema=AgentState,
)

response = graph.invoke({"messages": [HumanMessage("Hello")]})
print(response)
```

#### 5.2 Parallel Graph Orchestration

Use `create_parallel_pipeline` to orchestrate multiple subgraphs in parallel:

```python
from langchain_dev_utils.pipeline import create_parallel_pipeline

# Build a parallel pipeline (all subgraphs executed concurrently)
graph = create_parallel_pipeline(
    sub_graphs=[
        create_agent(
            model="vllm:qwen3-4b",
            tools=[get_current_time],
            system_prompt="You are a time-query assistant. You can only answer questions about the current time. If the question is unrelated to time, respond with 'I cannot answer that.'",
            name="time_agent",
        ),
        create_agent(
            model="vllm:qwen3-4b",
            tools=[get_current_weather],
            system_prompt="You are a weather-query assistant. You can only answer questions about the current weather. If the question is unrelated to weather, respond with 'I cannot answer that.'",
            name="weather_agent",
        ),
        create_agent(
            model="vllm:qwen3-4b",
            tools=[get_current_user],
            system_prompt="You are a user-query assistant. You can only answer questions about the current user. If the question is unrelated to the user, respond with 'I cannot answer that.'",
            name="user_agent",
        ),
    ],
    state_schema=AgentState,
)

response = graph.invoke({"messages": [HumanMessage("Hello")]})
print(response)
```

**For more information about state graph orchestration, please refer to**: [State Graph Orchestration](https://tbice123123.github.io/langchain-dev-utils-docs/en/graph-orchestration/pipeline.html)

## 💬 Join the Community

- [GitHub Repository](https://github.com/TBice123123/langchain-dev-utils) — Browse source code, submit Pull Requests
- [Issue Tracker](https://github.com/TBice123123/langchain-dev-utils/issues) — Report bugs or suggest improvements
- We welcome contributions in all forms — whether code, documentation, or usage examples. Let's build a more powerful and practical LangChain development ecosystem together!
