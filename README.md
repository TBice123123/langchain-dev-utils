# langchain-dev-utils

[![PyPI](https://img.shields.io/pypi/v/langchain-dev-utils.svg)](https://pypi.org/project/langchain-dev-utils/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/your-username/langchain-dev-utils/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)

**langchain-dev-utils** is a utility library focused on enhancing the development experience with LangChain and LangGraph. It provides a collection of ready-to-use utility functions that reduce repetitive code while improving code consistency and readability. By streamlining development workflows, this library helps you build prototypes faster, iterate more smoothly, and create clearer, more reliable AI applications powered by large language models.

## 📚 Documentation

- [English Documentation](https://tbice123123.github.io/langchain-dev-utils-docs/en/)
- [中文文档](https://tbice123123.github.io/langchain-dev-utils-docs/zh/)

## 🚀 Installation

```bash
pip install -U langchain-dev-utils

# For all features:
pip install -U langchain-dev-utils[standard]
```

## 📦 Core Features

### 1. **Model Management**

- Register any chat model or embeddings provider
- Unified interface with `load_chat_model()` / `load_embeddings()`

```python
# Chat model management
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

# Embeddings management
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

---

### 2. **Message Processing**

- Merge reasoning content into final response
- Stream-aware chunk merging
- Content formatting utilities

```python
from langchain_dev_utils.message_convert import (
    convert_reasoning_content_for_ai_message,
    convert_reasoning_content_for_chunk_iterator,
    merge_ai_message_chunk,
    format_sequence
)

response = model.invoke("Hello")
# Merge reasoning content to final response
cleaned = convert_reasoning_content_for_ai_message(
    response, think_tag=("<think>", "</think>")
)

# Stream merge reasoning content
for chunk in convert_reasoning_content_for_chunk_iterator(
    model.stream("Hello")
):
    print(chunk.content, end="", flush=True)

# Merge streaming chunks
chunks = list(model.stream("Hello"))
merged = merge_ai_message_chunk(chunks)

# Format sequence
text = format_sequence([
    "str1",
    "str2",
    "str3"
], separator="\n", with_num=True)
```

---

### 3. **Tool Calling**

- Check and parse tool calls
- Human-in-the-loop functionality for tool execution

```python
import datetime
from langchain_core.tools import tool
from langchain_dev_utils.tool_calling import has_tool_calling, parse_tool_calling, human_in_the_loop
from langchain_core.messages import AIMessage
from typing import cast

@human_in_the_loop
def get_current_time() -> str:
    """Get current timestamp"""
    return str(datetime.datetime.now().timestamp())

response = model.bind_tools([get_current_time]).invoke("What time is it?")

if has_tool_calling(cast(AIMessage, response)):
    name, args = parse_tool_calling(
        cast(AIMessage, response), first_tool_call_only=True
    )
    print(name, args)
```

---

### 4. **Agent Development**

- Pre-built agent factory functions
- Common middleware components

```python
# Basic agent
from langchain_dev_utils.agents import create_agent
from langchain.agents import AgentState

agent = create_agent("vllm:qwen3-4b", tools=[get_current_time], name="time-agent")
response = agent.invoke({"messages": [{"role": "user", "content": "What time is it?"}]})
print(response)


# Middleware
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
response = agent.invoke({"messages": [{"role": "user", "content": "Give me a plan to travel to New York"}]}))
print(response)
```

---

### 5. **State Graph Orchestration**

- Sequential graph pipelines
- Parallel graph pipelines

```python
from langchain_dev_utils.pipeline import sequential_pipeline, parallel_pipeline

# Build sequential pipeline
graph = sequential_pipeline(
    sub_graphs=[
        make_graph("graph1"),
        make_graph("graph2"),
        make_graph("graph3"),
    ],
    state_schema=State,
)

# Build parallel pipeline
graph = parallel_pipeline(
    sub_graphs=[
        make_graph("graph1"),
        make_graph("graph2"),
        make_graph("graph3"),
    ],
    state_schema=State,
)
```

---

## 💬 Join the Community

- 🐙 [GitHub Repository](https://github.com/TBice123123/langchain-dev-utils) — Browse source code, submit pull requests
- 🐞 [Issue Tracker](https://github.com/TBice123123/langchain-dev-utils/issues) — Report bugs or suggest improvements
- 💡 We welcome all forms of contribution — whether it's code, documentation, or usage examples. Let's build a more powerful and practical LangChain development ecosystem together!
