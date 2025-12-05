# Message Processing

> [!NOTE]
>
> **Feature Overview**: Provides practical message processing utilities.  
>
> **Prerequisites**: Familiarity with LangChain's [Message](https://docs.langchain.com/oss/python/langchain/messages).  
>
> **Estimated Reading Time**: 3 minutes

## Overview
Key functionalities include:
- Merging reasoning content into the final response
- Merging streamed output chunks

## Merging Reasoning Content into Final Response

These utilities merge reasoning content (`reasoning_content`) into the main response content (`content`).

Specifically:

- `convert_reasoning_content_for_ai_message`: Merges reasoning content from an `AIMessage` into its `content` field (for use with model `invoke` and `ainvoke`)
- `convert_reasoning_content_for_chunk_iterator`: Merges reasoning content from streamed responses into the `content` field (for use with model `stream`)
- `aconvert_reasoning_content_for_chunk_iterator`: An asynchronous version of `convert_reasoning_content_for_chunk_iterator`, for async streaming (for use with model `astream`)

### Parameters:

<Params  
name="model_response"  
type="AIMessage | Iterable[AIMessageChunk | AIMessage]"  
description="The AI message response returned by the model"  
:required="true"  
:default="null"  
/>

<Params  
name="think_tag"  
type="tuple[string, string]"  
description="A tuple containing start and end tags for wrapping reasoning content"  
:required="false"  
default="('\<think\>', '\</think\>')"  
/>

### Usage Example:

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

Provides utility functions to combine multiple `AIMessageChunk` objects—produced during streaming—into a single `AIMessage`.

Core function:

- `merge_ai_message_chunk`: Merges a sequence of AI message chunks

### Parameters:

<Params  
name="chunks"  
type="list[AIMessageChunk]"  
description="A list of AIMessageChunk objects"  
:required="true"  
:default="null"  
/>

### Usage Example:

```python
from langchain_dev_utils.message_convert import merge_ai_message_chunk

chunks = []
for chunk in model.stream("Hello"):
    chunks.append(chunk)

merged_message = merge_ai_message_chunk(chunks)
print(merged_message)
```