# langchain-dev-utils

[![PyPI](https://img.shields.io/pypi/v/langchain-dev-utils.svg)](https://pypi.org/project/langchain-dev-utils/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/your-username/langchain-dev-utils/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)

A practical enhancement utility library for LangChain / LangGraph developers, empowering the construction of complex and maintainable large language model applications.

## 📚 Documentation

- [中文文档](https://tbice123123.github.io/langchain-dev-utils-docs/zh/)
- [English Documentation](https://tbice123123.github.io/langchain-dev-utils-docs/en/)

## 🚀 Installation

```bash
pip install -U langchain-dev-utils

# For all features of this library:
pip install -U langchain-dev-utils[standard]
```

## 📦 Core Features

### 1. **Model Management**

- Supports registering any chat model or embedding model provider
- Provides unified interfaces `load_chat_model()` / `load_embeddings()` to simplify model loading
- Fully compatible with LangChain's official `init_chat_model` / `init_embeddings`, enabling seamless extension

```python
from langchain_dev_utils import register_model_provider, load_chat_model
from langchain_qwq import ChatQwen

register_model_provider("dashscope", ChatQwen)
register_model_provider("openrouter", "openai-compatible", base_url="https://openrouter.ai/api/v1")

model = load_chat_model("dashscope:qwen-flash")
print(model.invoke("Hello!"))
```

---

### 2. **Message Processing**

- Automatically merges reasoning content (e.g., from DeepSeek models) into the `content` field
- Supports streaming and asynchronous streaming responses (`stream` / `astream`)
- Utility functions include:
  - `merge_ai_message_chunk()`: merges message chunks
  - `has_tool_calling()` / `parse_tool_calling()`: detects and parses tool calls
  - `message_format()`: formats messages or document lists (with numbering, separators, etc.)

```python
from langchain_dev_utils import has_tool_calling, parse_tool_calling

response = model.invoke("What time is it now?")
if has_tool_calling(response):
    tool_calls = parse_tool_calling(response)
    print(tool_calls)
```

---

### 3. **Tool Enhancement**

- Easily extend existing tools with new capabilities
- Currently supports adding **human-in-the-loop** functionality to tools

```python
from langchain_dev_utils import human_in_the_loop_async
from langchain_core.tools import tool
import asyncio
import datetime

@human_in_the_loop_async
@tool
async def async_get_current_time() -> str:
    """Asynchronously retrieve the current timestamp"""
    await asyncio.sleep(1)
    return str(datetime.datetime.now().timestamp())
```

---

### 4. **Context Engineering**

- Automatically generates essential context management tools:
  - `create_write_plan_tool()` / `create_update_plan_tool()`
  - `create_write_note_tool()` / `create_query_note_tool()` / `create_ls_tool()` / `create_update_note_tool()`
- Provides corresponding State classes—no need to reimplement them

```python
from langchain_dev_utils import (
    create_write_plan_tool,
    create_update_plan_tool,
    create_write_note_tool,
    create_ls_tool,
    create_query_note_tool,
    create_update_note_tool,
)

plan_tools = [create_write_plan_tool(), create_update_plan_tool()]
note_tools = [create_write_note_tool(), create_ls_tool(), create_query_note_tool(), create_update_note_tool()]
```

---

### 5. **Graph Orchestration**

- Composes multiple `StateGraph`s in **sequential** or **parallel** fashion
- Supports complex multi-agent workflows:
  - `sequential_pipeline()`: executes subgraphs sequentially
  - `parallel_pipeline()`: executes subgraphs in parallel with dynamic branching (via the `Send` API)
- Allows specifying entry nodes and custom state/input/output schemas

```python
from langchain_dev_utils import parallel_pipeline, Send
from typing import TypedDict

class State(TypedDict):
    a: str
    results: list

def branches_fn(state: State):
    return [
        Send("graph1", arg={"a": state["a"]}),
        Send("graph2", arg={"a": state["a"]}),
    ]

graph = parallel_pipeline(
    sub_graphs=[graph1, graph2, graph3],
    state_schema=State,
    branches_fn=branches_fn,
)
```

---

### 6. **Prebuilt Agent**

This function provides functionality similar to `LangGraph`'s `create_react_agent`, but **only supports** string-based model parameters (loaded via `load_chat_model`), simplifying the model configuration process.

```python
from langchain_core.tools import tool
from langchain_dev_utils.prebuilt import create_agent
import datetime

@tool
def get_current_time() -> str:
    """Get current timestamp"""
    return str(datetime.datetime.now().timestamp())

# Only supports string format model identifiers
agent = create_agent(model="dashscope:qwen-flash", tools=[get_current_time])
response = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "What time is it now?"}]}
)
```

## 💬 Join the Community

- 🐙 [GitHub Repository](https://github.com/TBice123123/langchain-dev-utils) — Browse source code, submit pull requests
- 🐞 [Issue Tracker](https://github.com/TBice123123/langchain-dev-utils/issues) — Report bugs or suggest improvements
- 💡 We welcome contributions — whether it’s code, documentation, or usage examples. Help us build a stronger, more powerful ecosystem of practical langchain development tools!
