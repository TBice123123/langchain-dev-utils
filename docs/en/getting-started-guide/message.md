# Message Processing

## Overview
Main functions include:

- Merging reasoning content into final replies

- Merging streamed output Chunks

## Merging Reasoning Content into Final Replies

Used to merge reasoning content (`reasoning_content`) into the final reply (`content`).

Specifically:

- `convert_reasoning_content_for_ai_message`: Merges reasoning content in AIMessage into the content field (used for model's invoke and ainvoke)
- `convert_reasoning_content_for_chunk_iterator`: Merges reasoning content in streaming responses into the content field (used for model's stream)
- `aconvert_reasoning_content_for_chunk_iterator`: Asynchronous version of `convert_reasoning_content_for_chunk_iterator`, used for asynchronous streaming processing (used for model's astream)

Usage examples:

```python
from langchain_dev_utils.message_convert import (
    convert_reasoning_content_for_ai_message,
    convert_reasoning_content_for_chunk_iterator,
)

response = model.invoke("Hello")
converted_response = convert_reasoning_content_for_ai_message(
    response, think_tag=("<start>", "<end>")
)
print(converted_response.content)

for chunk in convert_reasoning_content_for_chunk_iterator(
    model.stream("Hello"), think_tag=("<start>", "<end>")
):
    print(chunk.content, end="", flush=True)
```

## Merging Streamed Output Chunks

Provides utility functions to merge multiple AIMessageChunks generated from streaming output into a single AIMessage.
Core function:

- `merge_ai_message_chunk`: Merges AI message chunks

Usage example:

```python
from langchain_dev_utils.message_convert import merge_ai_message_chunk

chunks = []
for chunk in model.stream("Hello"):
    chunks.append(chunk)

merged_message = merge_ai_message_chunk(chunks)
print(merged_message)
```