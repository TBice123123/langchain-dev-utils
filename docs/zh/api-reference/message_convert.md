# Message Convert 模块 API 参考文档

## convert_reasoning_content_for_ai_message

将思维链合并到最终回复中。

### 函数签名

```python
def convert_reasoning_content_for_ai_message(
    model_response: AIMessage,
    think_tag: Tuple[str, str] = ("<think>", "</think>"),
) -> AIMessage
```

### 参数

| 参数 | 类型 | 必填 | 默认值 | 描述 |
|------|------|------|--------|------|
| model_response | AIMessage | 是 | - | 包含推理内容的 AI 消息 |
| think_tag | Tuple[str, str] | 否 | `("<think>","</think>")` | 推理内容的开始和结束标签 |


### 示例

```python
response = convert_reasoning_content_for_ai_message(
    response, think_tag=("<start>", "<end>")
)
```

---

## convert_reasoning_content_for_chunk_iterator

为流式消息块合并推理内容。

### 函数签名

```python
def convert_reasoning_content_for_chunk_iterator(
    model_response: Iterator[AIMessageChunk | AIMessage],
    think_tag: Tuple[str, str] = ("<think>", "</think>"),
) -> Iterator[AIMessageChunk | AIMessage]
```

### 参数

| 参数 | 类型 | 必填 | 默认值 | 描述 |
|------|------|------|--------|------|
| model_response | Iterator[AIMessageChunk \| AIMessage] | 是 | - | 消息块的迭代器 |
| think_tag | Tuple[str, str] | 否 | `("<think>","</think>")` | 推理内容的开始和结束标签 |


### 示例

```python
for chunk in convert_reasoning_content_for_chunk_iterator(
    model.stream("Hello"), think_tag=("<think>", "</think>")
):
    print(chunk.content, end="", flush=True)
```

---

## aconvert_reasoning_content_for_chunk_iterator

`convert_reasoning_content_for_chunk_iterator` 的异步版本。

### 函数签名

```python
async def aconvert_reasoning_content_for_chunk_iterator(
    model_response: AsyncIterator[AIMessageChunk | AIMessage],
    think_tag: Tuple[str, str] = ("<think>", "</think>"),
) -> AsyncIterator[AIMessageChunk | AIMessage]
```

### 参数

| 参数 | 类型 | 必填 | 默认值 | 描述 |
|------|------|------|--------|------|
| model_response | AsyncIterator[AIMessageChunk \| AIMessage] | 是 | - | 消息块的异步迭代器 |
| think_tag | Tuple[str, str] | 否 | `("<think>","</think>")` | 推理内容的开始和结束标签 |


### 示例

```python
async for chunk in aconvert_reasoning_content_for_chunk_iterator(
    model.astream("Hello"), think_tag=("<think>", "</think>")
):
    print(chunk.content, end="", flush=True)
```

---

## merge_ai_message_chunk

将流式输出的 chunks 合并为一个 AIMessage。

### 函数签名

```python
def merge_ai_message_chunk(
    chunks: Sequence[AIMessageChunk]
) -> AIMessage
```

### 参数

| 参数 | 类型 | 必填 | 默认值 | 描述 |
|------|------|------|--------|------|
| chunks | Sequence[AIMessageChunk] | 是 | - | 待合并的消息块列表 |


### 示例

```python
chunks = list(model.stream("Hello"))
merged = merge_ai_message_chunk(chunks)
```

---

## format_sequence

将 BaseMessage、Document 或字符串列表格式化为单个字符串。

### 函数签名

```python
def format_sequence(
    inputs: List[Union[BaseMessage, Document, str]],
    separator: str = "-",
    with_num: bool = False
) -> str
```

### 参数

| 参数 | 类型 | 必填 | 默认值 | 描述 |
|------|------|------|--------|------|
| inputs | List[Union[BaseMessage, Document, str]] | 是 | - | 待格式化的项目列表 |
| separator | str | 否 | "-" | 分隔符字符串 |
| with_num | bool | 否 | False | 是否添加数字前缀 |



### 示例

```python
formatted = format_sequence(messages, separator="\n", with_num=True)
```