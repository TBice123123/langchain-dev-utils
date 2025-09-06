# LangChain Dev Utils

This toolkit is designed to provide encapsulated utility functions for developers building applications with large language models using LangChain and LangGraph, helping developers work more efficiently.

[中文文档](https://github.com/TBice123123/langchain-dev-utils/blob/master/README_cn.md)

## Installation and Usage

1. Using pip

```bash
pip install -U langchain-dev-utils
```

2. Using poetry

```bash
poetry add langchain-dev-utils
```

3. Using uv

```bash
uv add langchain-dev-utils
```

## Functional Modules

Currently divided into the following three main modules:

---

### 1. Instantiating Model Objects

While the official `init_chat_model` and `init_embeddings` functions are convenient to use, they support a relatively limited number of model providers. To address this, we provide `register_model_provider` and `register_embeddings_provider` functions. Through a unified registration and loading mechanism, developers can flexibly register any model provider, enabling broader model support. At the same time, `load_chat_model` and `load_embeddings` maintain the same simplicity in usage as the official functions.

#### (1) ChatModel Class

**Core Functions**

- `register_model_provider`: Register a model provider
- `load_chat_model`: Load a chat model

**Parameters for `register_model_provider`**

- `provider_name`: Provider name; must be a custom name
- `chat_model`: Either a ChatModel class or a string. If it's a string, it must be a provider supported by the official `init_chat_model` (e.g., `openai`, `anthropic`). In this case, the `init_chat_model` function will be called.
- `base_url`: Optional base URL. Recommended when `chat_model` is a string.

**Parameters for `load_chat_model`**

- `model`: Model name, in the format `model_name` or `provider_name:model_name`
- `model_provider`: Optional model provider name. If not provided, the provider name must be included in the `model` parameter.
- `kwargs`: Optional additional model parameters, such as `temperature`, `api_key`, `stop`, etc.  
  These three parameters are consistent with those of the official `init_chat_model` function.

- **Note**: Currently, passing `configurable_fields` and `config_prefix` parameters is not supported.

**Usage Example**

```python
from langchain_dev_utils import register_model_provider, load_chat_model
from langchain_qwq import ChatQwen
from dotenv import load_dotenv

load_dotenv()

# Register custom model providers
register_model_provider("dashscope", ChatQwen)
register_model_provider("openrouter", "openai", base_url="https://openrouter.ai/api/v1")

# Load models
model = load_chat_model(model="dashscope:qwen-flash")
print(model.invoke("Hello"))

model = load_chat_model(model="openrouter:moonshotai/kimi-k2-0905")
print(model.invoke("Hello"))
```

**Important**: Since the underlying implementation uses a global dictionary, **all model providers must be registered at application startup**. Modifications during runtime should be avoided to prevent multi-thread concurrency synchronization issues.

**Recommendation**: We suggest placing the `register_model_provider` calls in your application's `__init__.py` file.

For example, if you have the following LangGraph project structure:

```text
langgraph-project/
├── src
│   ├── __init__.py
│   └── graphs
│       ├── __init__.py # Call register_model_provider here
│       ├── graph1
│       └── graph2
```

#### (2) Embeddings Class

**Core Functions**

- `register_embeddings_provider`: Register an embeddings model provider
- `load_embeddings`: Load an embeddings model

**Parameters for `register_embeddings_provider`**

- `provider_name`: Provider name; must be a custom name
- `embeddings_model`: Either an Embeddings class or a string. If it's a string, it must be a provider supported by the official `init_embeddings` (e.g., `openai`, `anthropic`). In this case, the `init_embeddings` function will be called.
- `base_url`: Optional base URL. Recommended when `embeddings_model` is a string.

**Parameters for `load_embeddings`**

- `model`: Model name, in the format `model_name` or `provider_name:model_name`
- `provider`: Optional model provider name. If not provided, the provider name must be included in the `model` parameter.
- `kwargs`: Optional additional model parameters, such as `chunk_size`, `api_key`, `dimensions`, etc.  
  These three parameters are consistent with those of the official `init_embeddings` function.

**Usage Example**

```python
from langchain_dev_utils import register_embeddings_provider, load_embeddings

register_embeddings_provider(
    "dashscope", "openai", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

embeddings = load_embeddings("dashscope:text-embedding-v4")

print(embeddings.embed_query("hello world"))
```

**Important**: Similarly, since the underlying implementation uses a global dictionary, **all embedding model providers must be registered at application startup**, and no modifications should be made afterward to avoid multi-thread concurrency issues.

As before, we recommend placing `register_embeddings_provider` in your application's `__init__.py` file. Refer to the previous section on registering model providers for details.

---

### Message Class Processing

#### (1) Merging Reasoning Content

Provides functionality to merge the `reasoning_content` returned by reasoning models into the `content` field of AI messages.

**Core Functions**

- `convert_reasoning_content_for_ai_message`: Merge reasoning_content from AIMessage into content
- `convert_reasoning_content_for_chunk_iterator`: Merge reasoning_content for message chunk iterators in streaming responses
- `aconvert_reasoning_content_for_chunk_iterator`: Asynchronous version of `convert_reasoning_content_for_chunk_iterator` for async streaming

**Parameters**

- `model_response`: The AI message response from the model
- `think_tag`: A tuple containing the start and end tags for reasoning content (e.g., `("<think>", "</think>")`)

**Usage Example**

```python
# Synchronous processing of reasoning content
from typing import cast
from langchain_dev_utils import convert_reasoning_content_for_ai_message
from langchain_core.messages import AIMessage

# Streaming processing of reasoning content
from langchain_dev_utils import convert_reasoning_content_for_chunk_iterator

response = model.invoke("Hello")
converted_response = convert_reasoning_content_for_ai_message(
    cast(AIMessage, response), think_tag=("<!--THINK-->", "<!--/THINK-->")
)
print(converted_response.content)

for chunk in convert_reasoning_content_for_chunk_iterator(
    model.stream("Hello"), think_tag=("<!--THINK-->", "<!--/THINK-->")
):
    print(chunk.content, end="", flush=True)
```

#### (2) Merging AI Message Chunks

Provides utility functions to merge AI message chunks, combining multiple AI message chunks into a single AI message.

**Core Function**

- `merge_ai_message_chunk`: Merge AI message chunks

**Parameters**

- `chunks`: List of AI message chunks

**Usage Example**

```python
from langchain_dev_utils import merge_ai_message_chunk

chunks = []
for chunk in model.stream("Hello"):
    chunks.append(chunk)

merged_message = merge_ai_message_chunk(chunks)
print(merged_message)
```

#### (3) Detecting if Message Contains Tool Calls

Provides a simple function to detect whether a message contains tool calls.

**Core Function**

- `has_tool_calling`: Check if a message contains tool calls

**Parameters**

- `message`: An AIMessage object

**Usage Example**

```python
import datetime
from langchain_core.tools import tool
from langchain_dev_utils import has_tool_calling
from langchain_core.messages import AIMessage
from typing import cast

@tool
def get_current_time() -> str:
    """Get the current timestamp"""
    return str(datetime.datetime.now().timestamp())

response = model.bind_tools([get_current_time]).invoke("What is the current time?")
print(has_tool_calling(cast(AIMessage, response)))
```

#### (4) Parsing Tool Call Arguments

Provides a utility function to parse tool call arguments, extracting them from the message.

**Core Function**

- `parse_tool_calling`: Parse tool call arguments

**Parameters**

- `message`: An AIMessage object
- `first_tool_call_only`: Whether to parse only the first tool call. If `True`, returns a single tuple; if `False`, returns a list of tuples.

**Usage Example**

```python
import datetime
from langchain_core.tools import tool
from langchain_dev_utils import has_tool_calling, parse_tool_calling
from langchain_core.messages import AIMessage
from typing import cast

@tool
def get_current_time() -> str:
    """Get the current timestamp"""
    return str(datetime.datetime.now().timestamp())

response = model.bind_tools([get_current_time]).invoke("What is the current time?")

if has_tool_calling(cast(AIMessage, response)):
    name, args = parse_tool_calling(
        cast(AIMessage, response), first_tool_call_only=True
    )
    print(name, args)
```

#### (5) Formatting Messages

Formats a list composed of Documents, Messages, or strings into a single string.

**Core Function**

- `message_format`: Format messages

**Parameters**

- `inputs`: A list containing any of the following types:
  - `langchain_core.messages`: HumanMessage, AIMessage, SystemMessage, ToolMessage
  - `langchain_core.documents.Document`
  - `str`
- `separator`: String used to join content. Default is `"-"`.
- `with_num`: If `True`, adds a numbered prefix to each item (e.g., `"1. Hello"`). Default is `False`.

**Usage Example**

```python
from langchain_dev_utils import message_format
from langchain_core.documents import Document

messages = [
    Document(page_content="Document 1"),
    Document(page_content="Document 2"),
    Document(page_content="Document 3"),
    Document(page_content="Document 4"),
]
formatted_messages = message_format(messages, separator="\n", with_num=True)
print(formatted_messages)
```

---

### 3. Tool Enhancement

#### (1) Adding Interrupts to Tool Calls

Provides utility functions to add human-in-the-loop review support to tool calls, enabling human review during tool execution.

**Core Functions**

- `human_in_the_loop`: Add human-in-the-loop review to tool calls
- `human_in_the_loop_async`: Asynchronous version of `human_in_the_loop`

**Parameters**

- `func`: The function to be decorated. **Do not pass this parameter directly.**
- `interrupt_config`: Configuration for human interruption.

**Usage Example**

```python
from langchain_dev_utils import human_in_the_loop
from langchain_core.tools import tool
import datetime

@human_in_the_loop
@tool  # Can also be used without @tool
def get_current_time() -> str:
    """Get the current timestamp"""
    return str(datetime.datetime.now().timestamp())
```

## Testing

All utility functions in this project have been tested. You can also clone the repository to run the tests:

```bash
git clone https://github.com/TBice123123/langchain-dev-utils.git
```

```bash
cd langchain-dev-utils
```

```bash
uv sync --group test
```

```bash
uv run pytest .
```
