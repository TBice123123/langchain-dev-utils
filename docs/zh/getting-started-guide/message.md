# 消息处理

## 概述
主要功能包括：

- 合并推理内容至最终回复

- 合并流式输出的 Chunks

## 合并推理内容至最终回复

用于将推理内容（`reasoning_content`）合并至最终回复（`content`）。

具体如下

- `convert_reasoning_content_for_ai_message`：将 AIMessage 中的推理内容合并到内容字段（用于模型的 invoke 和 ainvoke）
- `convert_reasoning_content_for_chunk_iterator`：将流式响应中的推理内容合并到内容字段 (用于模型的 stream)
- `aconvert_reasoning_content_for_chunk_iterator`：`convert_reasoning_content_for_chunk_iterator` 的异步版本，用于异步流式处理(用于模型的 astream)


使用示例:

```python
from langchain_dev_utils.message_convert import (
    convert_reasoning_content_for_ai_message,
    convert_reasoning_content_for_chunk_iterator,
)

response = model.invoke("你好")
converted_response = convert_reasoning_content_for_ai_message(
    response, think_tag=("<start>", "<end>")
)
print(converted_response.content)

for chunk in convert_reasoning_content_for_chunk_iterator(
    model.stream("你好"), think_tag=("<start>", "<end>")
):
    print(chunk.content, end="", flush=True)
```

## 合并流式输出的 Chunks

提供将多个因为流式输出而产生的 AIMessageChunk 合并为单个 AIMessage 的工具函数。
核心函数为：

- `merge_ai_message_chunk`：合并 AI 消息块

使用示例:

```python
from langchain_dev_utils.message_convert import merge_ai_message_chunk

chunks = []
for chunk in model.stream("你好"):
    chunks.append(chunk)

merged_message = merge_ai_message_chunk(chunks)
print(merged_message)
```
