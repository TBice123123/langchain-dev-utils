# langchain-dev-utils

A practical enhancement utility library for LangChain / LangGraph developers, empowering the construction of complex and maintainable large language model applications.

## 🚀 Installation

```bash
pip install -U langchain-dev-utils
```

## 📦 Core Features

### 1. **Model Management**

- Supports registering any chat model or embedding model provider
- Provides unified interfaces `load_chat_model()` / `load_embeddings()` to simplify model loading
- Fully compatible with LangChain’s official `init_chat_model` / `init_embeddings`, enabling seamless extension

```python
from langchain_dev_utils import register_model_provider, load_chat_model
from langchain_qwq import ChatQwen

register_model_provider("dashscope", ChatQwen)
register_model_provider("openrouter", "openai", base_url="https://openrouter.ai/api/v1")

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
from langchain_dev_utils import parallel_pipeline

graph = parallel_pipeline(
    sub_graphs=[graph1, graph2, graph3],
    state_schema=State,
    branches_fn=lambda state: [
        Send("graph1", arg={"a": state["a"]}),
        Send("graph2", arg={"a": state["a"]}),
    ]
)
```

---

## 📚 Documentation and Examples

**For more information**, please refer to the following [documentation](https://tbice123123.github.io/langchain-dev-utils-docs/en/).
