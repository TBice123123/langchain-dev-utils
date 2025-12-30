# Message Convert Module API Reference Documentation

## convert_reasoning_content_for_ai_message

Merges the chain of thought into the final response.

### Function Signature

```python
def convert_reasoning_content_for_ai_message(
    model_response: AIMessage,
    think_tag: Tuple[str, str] = ("<think>", "</think>"),
) -> AIMessage
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| model_response | AIMessage | Yes | - | AI message containing reasoning content |
| think_tag | Tuple[str, str] | No | `("<think>","</think>")` | Start and end tags for reasoning content |

### Example

```python
response = convert_reasoning_content_for_ai_message(
    response, think_tag=("<start>", "<end>")
)
```

---

## convert_reasoning_content_for_chunk_iterator

Merges reasoning content for streaming message chunks.

### Function Signature

```python
def convert_reasoning_content_for_chunk_iterator(
    model_response: Iterator[AIMessageChunk | AIMessage],
    think_tag: Tuple[str, str] = ("think", "think"),
) -> Iterator[AIMessageChunk | AIMessage]
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| model_response | Iterator[AIMessageChunk \| AIMessage] | Yes | - | Iterator of message chunks |
| think_tag | Tuple[str, str] | No | `("<think>","</think>")` | Start and end tags for reasoning content |

### Example

```python
for chunk in convert_reasoning_content_for_chunk_iterator(
    model.stream("Hello"), think_tag=("<think>", "</think>")
):
    print(chunk.content, end="", flush=True)
```

---

## aconvert_reasoning_content_for_chunk_iterator

Asynchronous version of `convert_reasoning_content_for_chunk_iterator`.

### Function Signature

```python
async def aconvert_reasoning_content_for_chunk_iterator(
    model_response: AsyncIterator[AIMessageChunk | AIMessage],
    think_tag: Tuple[str, str] = ("think", "think"),
) -> AsyncIterator[AIMessageChunk | AIMessage]
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| model_response | AsyncIterator[AIMessageChunk \| AIMessage] | Yes | - | Async iterator of message chunks |
| think_tag | Tuple[str, str] | No | `("<think>","</think>")` | Start and end tags for reasoning content |

### Example

```python
async for chunk in aconvert_reasoning_content_for_chunk_iterator(
    model.astream("Hello"), think_tag=("<think>", "</think>")
):
    print(chunk.content, end="", flush=True)
```

---

## merge_ai_message_chunk

Merges streaming output chunks into a single AIMessage.

### Function Signature

```python
def merge_ai_message_chunk(
    chunks: Sequence[AIMessageChunk]
) -> AIMessage
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| chunks | Sequence[AIMessageChunk] | Yes | - | List of message chunks to merge |

### Example

```python
chunks = list(model.stream("Hello"))
merged = merge_ai_message_chunk(chunks)
```

---

## format_sequence

Formats a list of BaseMessage, Document, or strings into a single string.

### Function Signature

```python
def format_sequence(
    inputs: List[Union[BaseMessage, Document, str]],
    separator: str = "-",
    with_num: bool = False
) -> str
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| inputs | List[Union[BaseMessage, Document, str]] | Yes | - | List of items to format |
| separator | str | No | "-" | Separator string |
| with_num | bool | No | False | Whether to add number prefix |

### Example

```python
formatted = format_sequence(messages, separator="\n", with_num=True)
```