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

### 2. Reasoning Content Processing Functionality

Provides utility functions for processing model reasoning content, supporting both synchronous and asynchronous operations.

#### Core Functions

- `convert_reasoning_content_for_ai_message`: Convert reasoning content for a single AI message
- `convert_reasoning_content_for_chunk_iterator`: Convert reasoning content for streaming response message chunk iterator
- `aconvert_reasoning_content_for_ai_message`: Asynchronously convert reasoning content for a single AI message
- `aconvert_reasoning_content_for_chunk_iterator`: Asynchronously convert reasoning content for streaming response message chunk iterator

#### Usage Example

```python
# Synchronously process reasoning content
from langchain_dev_utils.content import convert_reasoning_content_for_ai_message

response = model.invoke("Please solve this math problem")
converted_response = convert_reasoning_content_for_ai_message(response, think_tag=("", ""))

# Stream processing reasoning content
from langchain_dev_utils.content import convert_reasoning_content_for_chunk_iterator

for chunk in convert_reasoning_content_for_chunk_iterator(model.stream("Please solve this math problem"), think_tag=("", "")):
    print(chunk.content, end="", flush=True)
```

### 3. Embeddings Model Loading Functionality

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
