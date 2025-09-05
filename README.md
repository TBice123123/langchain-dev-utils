# LangChain Dev Utils

[中文文档](https://github.com/TBice123123/langchain-dev-utils/blob/master/README_cn.md)

This toolkit is designed to provide encapsulated utility tools for developers using LangChain and LangGraph to develop large language model applications, helping developers work more efficiently.

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

## Function Modules

### 1. Extended Model Loading Functionality

While the official `init_chat_model` function is very useful, it has limited support for model providers. This toolkit provides extended model loading functionality that allows registration and use of more model providers.

#### Core Functions

- `register_model_provider`: Register a model provider
- `load_chat_model`: Load a chat model

#### `register_model_provider` Parameter Description

- `provider_name`: Provider name, requires a custom name
- `chat_model`: ChatModel class or string. If it's a string, it must be a provider supported by the official `init_chat_model` (e.g., `openai`, `anthropic`). In this case, the `init_chat_model` function will be called
- `base_url`: Optional base URL. Recommended when `chat_model` is a string

#### Usage Example

```python
from langchain_dev_utils.chat_model import register_model_provider, load_chat_model
from langchain_qwq import ChatQwen
from dotenv import load_dotenv

load_dotenv()

# Register custom model providers
register_model_provider("dashscope", ChatQwen)
register_model_provider("openrouter", "openai", base_url="https://openrouter.ai/api/v1")

# Load models
model = load_chat_model(model="dashscope:qwen-flash")
print(model.invoke("Hello!"))

model = load_chat_model(model="openrouter:moonshotai/kimi-k2-0905")
print(model.invoke("Hello!"))
```

**Note**: Since the underlying implementation of the function is a global dictionary, **all model providers must be registered at application startup**. Modifications should not be made at runtime, otherwise multi-threading concurrency synchronization issues may occur.

We recommend that you place `register_model_provider` in the `__init__.py` file of your application.

For example, if you have the following LangGraph directory structure:

```text
langgraph-project/
├── src
│   ├── __init__.py
│   └── graphs
│       ├── __init__.py # call register_model_provider here
│       ├── graph1
│       └── graph2
```

### 2. Merge Inference Content

Provides a function to merge `reasoning_content` returned by the model into the `content` of AI messages.

#### Core Functions

- `convert_reasoning_content_for_ai_message`: Convert the reasoning content of a single AI message
- `convert_reasoning_content_for_chunk_iterator`: Convert the reasoning content of an iterator of message blocks in a streaming response
- `aconvert_reasoning_content_for_chunk_iterator`: Asynchronously convert the reasoning content of an iterator of message blocks in a streaming response

#### Usage Example

```python
# 同步处理推理内容
from langchain_dev_utils.content import convert_reasoning_content_for_ai_message

response = model.invoke("请解决这个数学问题")
converted_response = convert_reasoning_content_for_ai_message(response, think_tag=("<think>", "</think>"))

# 流式处理推理内容
from langchain_dev_utils.content import convert_reasoning_content_for_chunk_iterator

for chunk in convert_reasoning_content_for_chunk_iterator(model.stream("请解决这个数学问题"), think_tag=("<think>", "</think>")):
    print(chunk.content, end="", flush=True)
```

### 3. Extended Embeddings Model Loading Functionality

Provides extended embeddings model loading functionality, similar to the model loading functionality.

#### Core Functions

- `register_embeddings_provider`: Register an embeddings model provider
- `load_embeddings`: Load an embeddings model

#### Usage Example

```python
from langchain_dev_utils.embbedings import register_embeddings_provider, load_embeddings

# Register embeddings model provider
register_embeddings_provider("openai", "openai", base_url="https://api.openai.com/v1")

# Load embeddings model
embeddings = load_embeddings("openai:text-embedding-ada-002")
```

**Note**: Since the underlying implementation of the function is a global dictionary, **all embeddings model providers must be registered at application startup**. Modifications should not be made at runtime, otherwise multi-threading concurrency synchronization issues may occur.

We recommend that you place `register_embeddings_provider` in the `__init__.py` file of your application.

### 4. Tool Calling Detection Functionality

Provides a simple function to detect whether a message contains tool calls.

#### Core Functions

- `has_tool_calling`: Detect whether a message contains tool calls

#### Usage Example

```python
from langchain_dev_utils.has_tool_calling import has_tool_calling

if has_tool_calling(message):
    # Handle tool calling logic
    pass
```

### 5. Merge AI Message Chunks

Provides a tool function for merging multiple AI message chunks into a single AI message.

#### Core Functions

- `merge_ai_message_chunk`: Merge AI message chunks

#### Usage Example

```python
from langchain_dev_utils.content import merge_ai_message_chunk

chunks = [
    AIMessageChunk(content="Chunk 1"),
    AIMessageChunk(content="Chunk 2"),
]
merged_message = merge_ai_message_chunk(chunks)
```

## Test

All the current tool functions in this project have been tested, and you can also clone this project for testing.

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
