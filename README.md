# langchain-dev-utils

[![PyPI](https://img.shields.io/pypi/v/langchain-dev-utils.svg)](https://pypi.org/project/langchain-dev-utils/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/your-username/langchain-dev-utils/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)

**langchain-dev-utils** is a practical utility library focused on enhancing the development experience with LangChain and LangGraph. It provides a series of out-of-the-box utility functions that not only reduce repetitive code writing but also improve code consistency and readability. By streamlining development workflows, this library helps you prototype faster, iterate more smoothly, and create clearer, more reliable LLM-based AI applications.

> Currently, English version. Chinese version please visit [中文文档](https://github.com/TBice123123/langchain-dev-utils/blob/master/README_cn.md)

## 📚 Documentation

- [English Documentation](https://tbice123123.github.io/langchain-dev-utils-docs/en/)
- [中文文档](https://tbice123123.github.io/langchain-dev-utils-docs/zh/)

## 🚀 Installation

```bash
pip install -U langchain-dev-utils

# Install the full-featured version:
pip install -U langchain-dev-utils[standard]
```

## 📦 Core Features

### 1. **Model Management**

In `langchain`, the `init_chat_model` function can be used to initialize a chat model instance, but the model providers it supports are relatively limited. This module provides a registration function (`register_model_provider`/`register_embeddings_provider`) to register any model provider for subsequent model loading using `load_chat_model` / `load_embeddings`.

`register_model_provider` parameters:

- `provider_name`: The model provider name, used as an identifier for subsequent model loading.
- `chat_model`: The chat model, which can be a ChatModel or a string (currently supports "openai-compatible").
- `base_url`: The API address of the model provider.

`register_embeddings_provider` parameters:

- `provider_name`: The embeddings model provider name, used as an identifier for subsequent model loading.
- `embeddings_model`: The embeddings model, which can be an Embeddings object or a string (currently supports "openai-compatible").
- `base_url`: The API address of the model provider.

Usage Example:

```python
# Chat Model Management
from langchain_dev_utils.chat_models import (
    register_model_provider,
    load_chat_model,
)

# Register a model provider
register_model_provider(
    provider_name="vllm",
    chat_model="openai-compatible",
    base_url="http://localhost:8000/v1",
)

# Load the model
model = load_chat_model("vllm:qwen3-4b")
print(model.invoke("Hello"))
```

Embeddings Model Usage:

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

**Learn More**: [Model Management](https://tbice123123.github.io/langchain-dev-utils-docs/en/model-management.html)

### 2. **Message Conversion**

Includes the following features:

- Merging reasoning content into the final response
- Streaming content merging
- Content formatting tools

Merging reasoning content into the final response:

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

Merging streaming responses:

```python
chunks = list(model.stream("Hello"))
merged = merge_ai_message_chunk(chunks)
```

Formatting sequences:

```python
text = format_sequence([
    "str1",
    "str2",
    "str3"
], separator="\n", with_num=True)
```

**Learn More**: [Message Conversion](https://tbice123123.github.io/langchain-dev-utils-docs/en/message-conversion.html)

### 3. **Tool Calling**

Includes the following features:

- Checking and parsing tool calls
- Adding human-in-the-loop functionality

Usage Example:

```python
import datetime
from langchain_core.tools import tool
from langchain_dev_utils.tool_calling import has_tool_calling, parse_tool_calling, human_in_the_loop
from langchain_core.messages import AIMessage
from typing import cast

@human_in_the_loop
def get_current_time() -> str:
    """Get the current timestamp"""
    return str(datetime.datetime.now().timestamp())

response = model.bind_tools([get_current_time]).invoke("What time is it now?")

if has_tool_calling(cast(AIMessage, response)):
    name, args = parse_tool_calling(
        cast(AIMessage, response), first_tool_call_only=True
    )
    print(name, args)
```

**Learn More**: [Tool Calling](https://tbice123123.github.io/langchain-dev-utils-docs/en/tool-calling.html)

### 4. **Agent Development**

Includes the following features:

- Predefined agent factory functions
- Common middleware components

Usage Example:

```python
from langchain_dev_utils.agents import create_agent
from langchain.agents import AgentState

agent = create_agent("vllm:qwen3-4b", tools=[get_current_time], name="time-agent")
response = agent.invoke({"messages": [{"role": "user", "content": "What time is it now?"}]})
print(response)
```

Middleware Usage:

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
response = agent.invoke({"messages": [{"role": "user", "content": "Give me a travel plan to New York"}]}))
print(response)
```

**Learn More**: [Agent Development](https://tbice123123.github.io/langchain-dev-utils-docs/en/agent-development.html)

### 5. **State Graph Orchestration**

Includes the following features:

- Sequential graph orchestration
- Parallel graph orchestration

```python
from langchain_dev_utils.pipeline import sequential_pipeline, parallel_pipeline

# Build a sequential workflow
graph = sequential_pipeline(
    sub_graphs=[
        make_graph("graph1"),
        make_graph("graph2"),
        make_graph("graph3"),
    ],
    state_schema=State,
)

# Build a parallel workflow
graph = parallel_pipeline(
    sub_graphs=[
        make_graph("graph1"),
        make_graph("graph2"),
        make_graph("graph3"),
    ],
    state_schema=State,
)
```

**Learn More**: [State Graph Orchestration](https://tbice123123.github.io/langchain-dev-utils-docs/en/graph-orchestration.html)

## 💬 Join the Community

- 🐙 [GitHub Repository](https://github.com/TBice123123/langchain-dev-utils) — Browse the source code, submit Pull Requests
- 🐞 [Issue Tracker](https://github.com/TBice123123/langchain-dev-utils/issues) — Report bugs or suggest improvements
- 💡 We welcome all forms of contribution — whether it's code, documentation, or usage examples. Let's build a more powerful and practical LangChain development ecosystem together!
