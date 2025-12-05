# Message Convert Module API Reference

## convert_reasoning_content_for_ai_message

Merges reasoning chain into the final response.

```python
def convert_reasoning_content_for_ai_message(
    model_response: AIMessage,
    think_tag: Tuple[str, str] = ("<think>", "</think>"),
) -> AIMessage
```

**Parameters:**

- `model_response`: AIMessage type, required, AI message containing reasoning content
- `think_tag`: String tuple type, optional, start and end tags for reasoning content, default `("<think>", "</think>")`

**Return Value:** AIMessage type, message with merged reasoning content

**Example:**

```python
response = convert_reasoning_content_for_ai_message(
    response, think_tag=("<start>", "<end>")
)
```

## convert_reasoning_content_for_chunk_iterator

Merges reasoning content for streaming message chunks.

```python
def convert_reasoning_content_for_chunk_iterator(
    model_response: Iterator[AIMessageChunk | AIMessage],
    think_tag: Tuple[str, str] = ("<think>", "</think>"),
) -> Iterator[AIMessageChunk | AIMessage]
```

**Parameters:**

- `model_response`: Iterator of AIMessageChunk or AIMessage type, required, iterator of message chunks
- `think_tag`: String tuple type, optional, start and end tags for reasoning content, default `("<think>", "</think>")`

**Return Value:** Iterator of AIMessageChunk or AIMessage type, processed message chunk iterator

**Example:**

```python
for chunk in convert_reasoning_content_for_chunk_iterator(
    model.stream("Hello"), think_tag=("<think>", "</think>")
):
    print(chunk.content, end="", flush=True)
```

## aconvert_reasoning_content_for_chunk_iterator

Asynchronous version of `convert_reasoning_content_for_chunk_iterator`.

```python
async def aconvert_reasoning_content_for_chunk_iterator(
    model_response: AsyncIterator[AIMessageChunk | AIMessage],
    think_tag: Tuple[str, str] = ("<think>", "</think>"),
) -> AsyncIterator[AIMessageChunk | AIMessage]
```

**Parameters:**

- `model_response`: AsyncIterator of AIMessageChunk or AIMessage type, required, asynchronous iterator of message chunks
- `think_tag`: String tuple type, optional, start and end tags for reasoning content, default `("<think>", "</think>")`

**Return Value:** AsyncIterator of BaseMessageChunk type, processed asynchronous message chunk iterator

**Example:**

```python
async for chunk in aconvert_reasoning_content_for_chunk_iterator(
    model.astream("Hello"), think_tag=("<think>", "</think>")
):
    print(chunk.content, end="", flush=True)
```

## merge_ai_message_chunk

Merges streaming output chunks into a single AIMessage.

```python
def merge_ai_message_chunk(
    chunks: Sequence[AIMessageChunk]
) -> AIMessage
```

**Parameters:**

- `chunks`: Sequence of AIMessageChunk type, required, list of message chunks to merge

**Return Value:** AIMessage type, merged message

**Example:**

```python
chunks = list(model.stream("Hello"))
merged = merge_ai_message_chunk(chunks)
```

## format_sequence

Formats a list of BaseMessage, Document, or strings into a single string.

```python
def format_sequence(
    inputs: List[Union[BaseMessage, Document, str]],
    separator: str = "-",
    with_num: bool = False
) -> str
```

**Parameters:**

- `inputs`: List of Message, Document, or strings type, required, list of items to format
- `separator`: String type, optional, separator string, default "-"
- `with_num`: Boolean type, optional, whether to add numeric prefix, default False

**Return Value:** String type, formatted string

**Example:**

```python
formatted = message_format(messages, separator="\n", with_num=True)
```
