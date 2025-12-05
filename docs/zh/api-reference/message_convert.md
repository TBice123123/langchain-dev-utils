# Message Convert 模块的 API 参考

## convert_reasoning_content_for_ai_message

将思维链合并到最终回复中。

```python
def convert_reasoning_content_for_ai_message(
    model_response: AIMessage,
    think_tag: Tuple[str, str] = ("<think>", "</think>"),
) -> AIMessage
```

**参数说明：**

- `model_response`：AIMessage 类型，必填，包含推理内容的 AI 消息
- `think_tag`：字符串元组类型，可选，推理内容的开始和结束标签，默认 `("<think>", "</think>")`

**返回值：** AIMessage 类型，合并推理内容后的消息

**示例：**

```python
response = convert_reasoning_content_for_ai_message(
    response, think_tag=("<start>", "<end>")
)
```

## convert_reasoning_content_for_chunk_iterator

为流式消息块合并推理内容。

```python
def convert_reasoning_content_for_chunk_iterator(
    model_response: Iterator[AIMessageChunk | AIMessage],
    think_tag: Tuple[str, str] = ("<think>", "</think>"),
) -> Iterator[AIMessageChunk | AIMessage]
```

**参数说明：**

- `model_response`：AIMessageChunk 或 AIMessage 迭代器类型，必填，消息块的迭代器
- `think_tag`：字符串元组类型，可选，推理内容的开始和结束标签，默认 `("<think>", "</think>")`

**返回值：** AIMessageChunk 或 AIMessage 迭代器类型，处理后的消息块迭代器

**示例：**

```python
for chunk in convert_reasoning_content_for_chunk_iterator(
    model.stream("Hello"), think_tag=("<think>", "</think>")
):
    print(chunk.content, end="", flush=True)
```

## aconvert_reasoning_content_for_chunk_iterator

`convert_reasoning_content_for_chunk_iterator` 的异步版本。

```python
async def aconvert_reasoning_content_for_chunk_iterator(
    model_response: AsyncIterator[AIMessageChunk | AIMessage],
    think_tag: Tuple[str, str] = ("<think>", "</think>"),
) -> AsyncIterator[AIMessageChunk | AIMessage]
```

**参数说明：**

- `model_response`：AIMessageChunk 或 AIMessage 异步迭代器类型，必填，消息块的异步迭代器
- `think_tag`：字符串元组类型，可选，推理内容的开始和结束标签，默认 `("<think>", "</think>")`

**返回值：** BaseMessageChunk 异步迭代器类型，处理后的异步消息块迭代器

**示例：**

```python
async for chunk in aconvert_reasoning_content_for_chunk_iterator(
    model.astream("Hello"), think_tag=("<think>", "</think>")
):
    print(chunk.content, end="", flush=True)
```

## merge_ai_message_chunk

将流式输出的 chunks 合并为一个 AIMessage。

```python
def merge_ai_message_chunk(
    chunks: Sequence[AIMessageChunk]
) -> AIMessage
```

**参数说明：**

- `chunks`：AIMessageChunk 序列类型，必填，待合并的消息块列表

**返回值：** AIMessage 类型，合并后的消息

**示例：**

```python
chunks = list(model.stream("Hello"))
merged = merge_ai_message_chunk(chunks)
```

## format_sequence

将 BaseMessage、Document 或字符串列表格式化为单个字符串。

```python
def format_sequence(
    inputs: List[Union[BaseMessage, Document, str]],
    separator: str = "-",
    with_num: bool = False
) -> str
```

**参数说明：**

- `inputs`：Message、Document 或字符串的列表类型，必填，待格式化的项目列表
- `separator`：字符串类型，可选，分隔符字符串，默认 "-"
- `with_num`：布尔类型，可选，是否添加数字前缀，默认 False

**返回值：** 字符串类型，格式化后的字符串

**示例：**

```python
formatted = message_format(messages, separator="\n", with_num=True)
```
