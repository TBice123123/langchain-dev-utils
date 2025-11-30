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

**`register_model_provider` Parameters:**

| Parameter               | Type             | Required | Default | Description                                                                                                                                                                                                                                                                     |
| ----------------------- | ---------------- | -------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `provider_name`         | str              | Yes      | -       | The name of the model provider, used as an identifier for loading models later.                                                                                                                                                                                                 |
| `chat_model`            | ChatModel \| str | Yes      | -       | The chat model, which can be either a `ChatModel` instance or a string (currently only `"openai-compatible"` is supported).                                                                                                                                                     |
| `base_url`              | str              | No       | -       | The API endpoint URL of the model provider (applicable to both `chat_model` types, but primarily used when `chat_model` is a string with value `"openai-compatible"`).                                                                                                          |
| `model_profiles`        | dict             | No       | -       | Declares the capabilities and parameters supported by each model provided by this provider. The configuration corresponding to the `model_name` will be loaded and assigned to `model.profile` (e.g., fields such as `max_input_tokens`, `tool_calling` etc.).                  |
| `compatibility_options` | dict             | No       | -       | Compatibility options for the model provider (only effective when `chat_model` is a string with value `"openai-compatible"`). Used to declare support for OpenAI-compatible features (e.g., `tool_choice` strategies, JSON mode, etc.) to ensure correct functional adaptation. |

**`load_chat_model` Parameters:**

| Parameter        | Type | Required | Default | Description                                                                          |
| ---------------- | ---- | -------- | ------- | ------------------------------------------------------------------------------------ |
| `model`          | str  | Yes      | -       | Chat model name                                                                      |
| `model_provider` | str  | No       | -       | Chat model provider name                                                             |
| `kwargs`         | dict | No       | -       | Additional parameters passed to the chat model class, e.g., temperature, top_p, etc. |

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

**`register_embeddings_provider` Parameters:**

| Parameter          | Type              | Required | Default | Description                                                                                                                                                                  |
| ------------------ | ----------------- | -------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `provider_name`    | str               | Yes      | -       | Embedding model provider name, used as an identifier for subsequent model loading                                                                                            |
| `embeddings_model` | Embeddings \| str | Yes      | -       | Embedding model, can be Embeddings or a string (currently supports "openai-compatible")                                                                                      |
| `base_url`         | str               | No       | -       | The API address of the Embedding model provider (valid for both types of `embeddings_model`, but mainly used when `embeddings_model` is a string and is "openai-compatible") |

**`load_embeddings` Parameters:**

| Parameter  | Type | Required | Default | Description                   |
| ---------- | ---- | -------- | ------- | ----------------------------- |
| `model`    | str  | Yes      | -       | Embedding model name          |
| `provider` | str  | No       | -       | Embedding model provider name |
| `kwargs`   | dict | No       | -       | Other additional parameters   |

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

**`merge_ai_message_chunk` Parameters:**

| Parameter | Type                 | Required | Default | Description                    |
| --------- | -------------------- | -------- | ------- | ------------------------------ |
| `chunks`  | List[AIMessageChunk] | Yes      | -       | List of AIMessageChunk objects |

```python
from langchain_dev_utils.message_convert import merge_ai_message_chunk

chunks = list(model.stream("Hello"))
merged = merge_ai_message_chunk(chunks)
```

#### 2.2 Format List Content

For a list, you can use `format_sequence` to format it.

**`format_sequence` Parameters:**

| Parameter   | Type | Required | Default | Description                                                                                                   |
| ----------- | ---- | -------- | ------- | ------------------------------------------------------------------------------------------------------------- |
| `inputs`    | List | Yes      | -       | A list containing any of the following types: langchain_core.messages, langchain_core.documents.Document, str |
| `separator` | str  | No       | "-"     | String used to join the content                                                                               |
| `with_num`  | bool | No       | False   | If True, add a numeric prefix to each item (e.g., "1. Hello")                                                 |

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

**`has_tool_calling` Parameters:**

| Parameter | Type      | Required | Default | Description      |
| --------- | --------- | -------- | ------- | ---------------- |
| `message` | AIMessage | Yes      | -       | AIMessage object |

**`parse_tool_calling` Parameters:**

| Parameter              | Type      | Required | Default | Description                               |
| ---------------------- | --------- | -------- | ------- | ----------------------------------------- |
| `message`              | AIMessage | Yes      | -       | AIMessage object                          |
| `first_tool_call_only` | bool      | No       | False   | Whether to only parse the first tool call |

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

Includes the following features:

- Predefined agent factory functions
- Common middleware components

#### 4.1 Agent Factory Functions

In LangChain v1, the officially provided `create_agent` function can be used to create a single agent, where the model parameter supports passing a BaseChatModel instance or a specific string (when passing a string, it is limited to the models supported by `init_chat_model`). To extend the flexibility of specifying models via strings, this library provides a functionally identical `create_agent` function, allowing you to directly use models supported by `load_chat_model` (requires prior registration).

**`create_agent` Parameters:**

| Parameter        | Type    | Required | Default | Description                                                                                                                                 |
| ---------------- | ------- | -------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`          | str     | Yes      | -       | Model name or model instance. Can be a string identifier for a model registered with `register_model_provider` or a BaseChatModel instance. |
| Other parameters | Various | No       | -       | All other parameters are the same as in `langchain.agents.create_agent`                                                                     |

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

**For more information about agent development and all built-in middleware, please refer to**: [Pre-built Agent Functions](https://tbice123123.github.io/langchain-dev-utils-docs/en/agent-development/prebuilt.html), [Middleware](https://tbice123123.github.io/langchain-dev-utils-docs/en/agent-development/middleware.html)

### 5. **State Graph Orchestration**

Includes the following features:

- Sequential graph orchestration
- Parallel graph orchestration

#### 5.1 Sequential Graph Orchestration

Sequential graph orchestration:
Uses `create_sequential_pipeline`, supported parameters:

**`create_sequential_pipeline` Parameters:**

| Parameter        | Type                                 | Required | Default | Description                                                                                    |
| ---------------- | ------------------------------------ | -------- | ------- | ---------------------------------------------------------------------------------------------- |
| `sub_graphs`     | List[StateGraph\|CompiledStateGraph] | Yes      | -       | List of state graphs to combine (must be StateGraph instances or CompiledStateGraph instances) |
| `state_schema`   | type[dict]                           | Yes      | -       | State Schema for the final generated graph                                                     |
| `graph_name`     | str                                  | No       | -       | Name of the final generated graph                                                              |
| `context_schema` | type[dict]                           | No       | -       | Context Schema for the final generated graph                                                   |
| `input_schema`   | type[dict]                           | No       | -       | Input Schema for the final generated graph                                                     |
| `output_schema`  | type[dict]                           | No       | -       | Output Schema for the final generated graph                                                    |
| `checkpoint`     | BaseCheckpointSaver                  | No       | -       | LangGraph persistence Checkpoint                                                               |
| `store`          | BaseStore                            | No       | -       | LangGraph persistence Store                                                                    |
| `cache`          | BaseCache                            | No       | -       | LangGraph Cache                                                                                |

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

# Build sequential pipeline (all sub-graphs execute sequentially)
graph = create_sequential_pipeline(
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
Uses `create_parallel_pipeline`, supported parameters:

**`create_parallel_pipeline` Parameters:**

| Parameter        | Type                                 | Required | Default | Description                                                                                    |
| ---------------- | ------------------------------------ | -------- | ------- | ---------------------------------------------------------------------------------------------- |
| `sub_graphs`     | List[StateGraph\|CompiledStateGraph] | Yes      | -       | List of state graphs to combine (must be StateGraph instances or CompiledStateGraph instances) |
| `state_schema`   | type[dict]                           | Yes      | -       | State Schema for the final generated graph                                                     |
| `branches_fn`    | Callable                             | No       | -       | Parallel branch function, returns a list of Send objects to control parallel execution         |
| `graph_name`     | str                                  | No       | -       | Name of the final generated graph                                                              |
| `context_schema` | type[dict]                           | No       | -       | Context Schema for the final generated graph                                                   |
| `input_schema`   | type[dict]                           | No       | -       | Input Schema for the final generated graph                                                     |
| `output_schema`  | type[dict]                           | No       | -       | Output Schema for the final generated graph                                                    |
| `checkpoint`     | BaseCheckpointSaver                  | No       | -       | LangGraph persistence Checkpoint                                                               |
| `store`          | BaseStore                            | No       | -       | LangGraph persistence Store                                                                    |
| `cache`          | BaseCache                            | No       | -       | LangGraph Cache                                                                                |

```python
from langchain_dev_utils.pipeline import create_parallel_pipeline

# Build parallel pipeline (all sub-graphs execute in parallel)
graph = create_parallel_pipeline(
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

**For more information about state graph orchestration, please refer to**: [State Graph Orchestration](https://tbice123123.github.io/langchain-dev-utils-docs/en/graph-orchestration/pipeline.html)

## 💬 Join the Community

- [GitHub Repository](https://github.com/TBice123123/langchain-dev-utils) — Browse source code, submit Pull Requests
- [Issue Tracker](https://github.com/TBice123123/langchain-dev-utils/issues) — Report bugs or suggest improvements
- We welcome contributions in all forms — whether code, documentation, or usage examples. Let's build a more powerful and practical LangChain development ecosystem together!
