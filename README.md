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
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://tbice123123.github.io/langchain-dev-utils-docs/zh/)

> This is the Chinese version. For the English version, please visit [English Documentation](https://github.com/TBice123123/langchain-dev-utils/blob/master/README.md)

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

`register_model_provider` parameter description:

- `provider_name`: Model provider name, used as an identifier for subsequent model loading
- `chat_model`: Chat model, can be a ChatModel or a string (currently supports "openai-compatible")
- `base_url`: API address of the model provider (optional, valid when `chat_model` is a string and is "openai-compatible")
- `provider_config`: Relevant configuration for the model provider (optional, valid when `chat_model` is a string and is "openai-compatible"), can configure some provider-related parameters, such as whether to support structured output in json_mode, list of supported tool_choices, etc.

`load_chat_model` parameter description:

- `model`: Chat model name, type str
- `model_provider`: Chat model provider name, type str, optional
- `kwargs`: Additional parameters passed to the chat model class, e.g., temperature, top_p, etc.

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

`register_embeddings_provider` parameter description:

- `provider_name`: Embedding model provider name, used as an identifier for subsequent model loading
- `embeddings_model`: Embedding model, can be Embeddings or a string (currently supports "openai-compatible")
- `base_url`: API address of the model provider (optional, valid when `embeddings_model` is a string and is "openai-compatible")

`load_embeddings` parameter description:

- `model`: Embedding model name, type str
- `provider`: Embedding model provider name, type str, optional
- `kwargs`: Other additional parameters

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

`merge_ai_message_chunk` parameter description:

- `chunks`: List of AIMessageChunk

```python
chunks = list(model.stream("Hello"))
merged = merge_ai_message_chunk(chunks)
```

#### 2.2 Format List Content

For a list, you can use `format_sequence` to format it.

`format_sequence` parameter description:

- `inputs`: A list containing any of the following types:
  - langchain_core.messages: HumanMessage, AIMessage, SystemMessage, ToolMessage
  - langchain_core.documents.Document
  - str
- `separator`: String used to join the content, defaults to "-".
- `with_num`: If True, add a numeric prefix to each item (e.g., "1. Hello"), defaults to False.

```python
text = format_sequence([
    "str1",
    "str2",
    "str3"
], separator="\n", with_num=True)
```

**For more information about message conversion, please refer to**: [Message Processing](https://tbice123123.github.io/langchain-dev-utils-docs/en/message-conversion/message.html), [Format List Content](https://tbice123123.github.io/langchain-dev-utils-docs/en/message-conversion/format.html)

### 3. **Tool Calling**

Includes the following features:

- Check and parse tool calls
- Add human-in-the-loop functionality

#### 3.1 Check and Parse Tool Calls

`has_tool_calling` and `parse_tool_calling` are used to check and parse tool calls.

`has_tool_calling` parameter description:

- `message`: AIMessage object

`parse_tool_calling` parameter description:

- `message`: AIMessage object
- `first_tool_call_only`: Whether to only check the first tool call

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
from langchain_dev_utils import human_in_the_loop
from langchain_core.tools import tool
import datetime

@human_in_the_loop
@tool
def get_current_time() -> str:
    """Get the current timestamp"""
    return str(datetime.datetime.now().timestamp())
```

**For more information about tool calling, please refer to**: [Add Human-in-the-Loop Support](https://tbice123123.github.io/langchain-dev-utils-docs/en/tool-calling/human-in-the-loop.html), [Tool Call Processing](https://tbice123123.github.io/langchain-dev-utils-docs/en/tool-calling/tool.html)

### 4. **Agent Development**

Includes the following features:

- Predefined agent factory functions
- Common middleware components

#### 4.1 Agent Factory Functions

In LangChain v1, the officially provided `create_agent` function can be used to create a single agent, where the model parameter supports passing a BaseChatModel instance or a specific string (when passing a string, it is limited to the models supported by `init_chat_model`). To extend the flexibility of specifying models via strings, this library provides a functionally identical `create_agent` function, allowing you to directly use models supported by `load_chat_model` (requires prior registration).

Usage example:

```python
from langchain_dev_utils.agents import create_agent
from langchain.agents import AgentState

agent = create_agent("vllm:qwen3-4b", tools=[get_current_time], name="time-agent")
response = agent.invoke({"messages": [{"role": "user", "content": "What time is it?"}]})
print(response)
```

#### 4.2 Middleware

Provides some commonly used middleware components. Below are examples of `SummarizationMiddleware` and `PlanMiddleware`.

`SummarizationMiddleware` is used for agent summarization.

`PlanMiddleware` is used for agent planning.

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
response = agent.invoke({"messages": [{"role": "user", "content": "Give me a travel plan to New York"}]}))
print(response)
```

**For more information about agent development and all built-in middleware, please refer to**: [Prebuilt Agent Functions](https://tbice123123.github.io/langchain-dev-utils-docs/en/agent-development/prebuilt.html), [Middleware](https://tbice123123.github.io/langchain-dev-utils-docs/en/agent-development/middleware.html)

### 5. **State Graph Orchestration**

Includes the following features:

- Sequential graph orchestration
- Parallel graph orchestration

#### 5.1 Sequential Graph Orchestration

Sequential graph orchestration:
Uses `sequential_pipeline`, supported parameters:

- `sub_graphs`: List of state graphs to combine (must be StateGraph instances)
- `state_schema`: State Schema for the final generated graph
- `graph_name`: Name of the final generated graph (optional)
- `context_schema`: Context Schema for the final generated graph (optional)
- `input_schema`: Input Schema for the final generated graph (optional)
- `output_schema`: Output Schema for the final generated graph (optional)
- `checkpoint`: LangGraph persistence Checkpoint (optional)
- `store`: LangGraph persistence Store (optional)
- `cache`: LangGraph Cache (optional)

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

# Build sequential pipeline (all sub-graphs execute sequentially)
graph = sequential_pipeline(
    sub_graphs=[
        create_agent(
            model="vllm:qwen3-4b",
            tools=[get_current_time],
            system_prompt="You are a time query assistant, can only answer the current time. If the question is unrelated to time, please directly answer that you cannot answer.",
            name="time_agent",
        ),
        create_agent(
            model="vllm:qwen3-4b",
            tools=[get_current_weather],
            system_prompt="You are a weather query assistant, can only answer the current weather. If the question is unrelated to weather, please directly answer that you cannot answer.",
            name="weather_agent",
        ),
        create_agent(
            model="vllm:qwen3-4b",
            tools=[get_current_user],
            system_prompt="You are a user query assistant, can only answer the current user. If the question is unrelated to user, please directly answer that you cannot answer.",
            name="user_agent",
        ),
    ],
    state_schema=AgentState,
)

response = graph.invoke({"messages": [HumanMessage("Hello")]})
print(response)
```

#### 5.2 Parallel Graph Orchestration

Parallel graph orchestration:
Uses `parallel_pipeline`, supported parameters:

- `sub_graphs`: List of state graphs to combine
- `state_schema`: State Schema for the final generated graph
- `branches_fn`: Parallel branch function, returns a list of Send objects to control parallel execution
- `graph_name`: Name of the final generated graph (optional)
- `context_schema`: Context Schema for the final generated graph (optional)
- `input_schema`: Input Schema for the final generated graph (optional)
- `output_schema`: Output Schema for the final generated graph (optional)
- `checkpoint`: LangGraph persistence Checkpoint (optional)
- `store`: LangGraph persistence Store (optional)
- `cache`: LangGraph Cache (optional)

```python
from langchain_dev_utils.pipeline import parallel_pipeline

# Build parallel pipeline (all sub-graphs execute in parallel)
graph = parallel_pipeline(
    sub_graphs=[
        create_agent(
            model="vllm:qwen3-4b",
            tools=[get_current_time],
            system_prompt="You are a time query assistant, can only answer the current time. If the question is unrelated to time, please directly answer that you cannot answer.",
            name="time_agent",
        ),
        create_agent(
            model="vllm:qwen3-4b",
            tools=[get_current_weather],
            system_prompt="You are a weather query assistant, can only answer the current weather. If the question is unrelated to weather, please directly answer that you cannot answer.",
            name="weather_agent",
        ),
        create_agent(
            model="vllm:qwen3-4b",
            tools=[get_current_user],
            system_prompt="You are a user query assistant, can only answer the current user. If the question is unrelated to user, please directly answer that you cannot answer.",
            name="user_agent",
        ),
    ],
    state_schema=AgentState,
)
response = graph.invoke({"messages": [HumanMessage("Hello")]})
print(response)
```

**For more information about state graph orchestration, please refer to**: [State Graph Orchestration Pipeline](https://tbice123123.github.io/langchain-dev-utils-docs/en/graph-orchestration/pipeline.html)

## 💬 Join the Community

- [GitHub Repository](https://github.com/TBice123123/langchain-dev-utils) — Browse source code, submit Pull Requests
- [Issue Tracker](https://github.com/TBice123123/langchain-dev-utils/issues) — Report bugs or suggest improvements
- We welcome contributions in all forms — whether code, documentation, or usage examples. Let's build a more powerful and practical LangChain development ecosystem together!
