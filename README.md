# LangChain Dev Utils

This toolkit is designed to provide encapsulated utility functions for developers building applications with large language models using LangChain and LangGraph, helping developers work more efficiently.

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

## Usage

### 1. Model Management ⭐

(1) ChatModel

```Python
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

model = load_chat_model(model="openrouter:moonshotai/kimi-k2-0905", temperature=0.7)
print(model.invoke("Hello"))

```

(2) Embedding

```python
from langchain_dev_utils import register_embeddings_provider, load_embeddings
from langchain_siliconflow import SiliconFlowEmbeddings

register_embeddings_provider(
    "dashscope", "openai", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

register_embeddings_provider("siliconflow", SiliconFlowEmbeddings)

embeddings = load_embeddings("dashscope:text-embedding-v4")
print(embeddings.embed_query("hello world"))

embeddings = load_embeddings("siliconflow:BAAI/bge-m3")
print(embeddings.embed_query("hello world"))
```

### 2. Message Utilities

```Python
from langchain_dev_utils import (
    convert_reasoning_content_for_ai_message,
    convert_reasoning_content_for_chunk_iterator,
    aconvert_reasoning_content_for_chunk_iterator,
    merge_ai_message_chunk,
    has_tool_calling,
    parse_tool_calling,
    message_format
)

# merge reasoning tags into content
msg = convert_reasoning_content_for_ai_message(ai_msg, think_tag=("<think>","</think>"))

# streaming (sync / async)
for chunk in convert_reasoning_content_for_chunk_iterator(model.stream("hi")):
    print(chunk.content, end="")

# re-assemble chunks
full = merge_ai_message_chunk(chunks)

# tool-call helpers
if has_tool_calling(msg):
    name, args = parse_tool_calling(msg, first_tool_call_only=True)

# pretty print mixed items
text = message_format(["text", "image", "note"], separator="\n", with_num=True)
```

### 3. Tool Enhancement

```Python
from langchain_dev_utils import human_in_the_loop, human_in_the_loop_async

@human_in_the_loop          # sync
@tool
def danger(x: int) -> str: ...

@human_in_the_loop_async    # async
@tool
async def danger_async(x: int) -> str: ...
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

**For more information**, please refer to the following [documents](https://tbice123123.github.io/langchain-dev-utils-docs/en/).
