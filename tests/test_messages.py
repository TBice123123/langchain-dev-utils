from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk, ToolCall
import pytest

from langchain_dev_utils import (
    aconvert_reasoning_content_for_chunk_iterator,
    convert_reasoning_content_for_ai_message,
    convert_reasoning_content_for_chunk_iterator,
    has_tool_calling,
    merge_ai_message_chunk,
    message_format,
    parse_tool_calling,
)


def test_convert_reasoning_content_for_ai_message():
    ai_message = AIMessage(
        content="Hello",
        additional_kwargs={"reasoning_content": "I think therefore I am"},
    )

    result = convert_reasoning_content_for_ai_message(
        ai_message, ("<think>", "</think>")
    )
    assert result.content == "<think>I think therefore I am</think>Hello"

    ai_message = AIMessage(
        content="Hello",
        additional_kwargs={"reasoning_content": "I think therefore I am"},
    )
    result = convert_reasoning_content_for_ai_message(ai_message, ("<", ">"))
    assert result.content == "<I think therefore I am>Hello"


def test_convert_reasoning_content_for_chunk_iterator():
    chunks = [
        AIMessageChunk(
            content="", additional_kwargs={"reasoning_content": "First thought"}
        ),
        AIMessageChunk(
            content="", additional_kwargs={"reasoning_content": "Second thought"}
        ),
        AIMessageChunk(content="Final answer"),
    ]

    result_chunks = list(
        convert_reasoning_content_for_chunk_iterator(
            iter(chunks), ("<think>", "</think>")
        )
    )

    assert result_chunks[0].content == "<think>First thought"
    assert result_chunks[1].content == "Second thought"
    assert result_chunks[2].content == "</think>Final answer"


@pytest.mark.asyncio
async def test_aconvert_reasoning_content_for_chunk_iterator():
    async def async_chunk_generator():
        chunks = [
            AIMessageChunk(
                content="",
                additional_kwargs={"reasoning_content": "First thought"},
            ),
            AIMessageChunk(
                content="",
                additional_kwargs={"reasoning_content": "Second thought"},
            ),
            AIMessageChunk(content="Final answer"),
        ]
        for chunk in chunks:
            yield chunk

    result_chunks = []
    async for chunk in aconvert_reasoning_content_for_chunk_iterator(
        async_chunk_generator(), ("<think>", "</think>")
    ):
        result_chunks.append(chunk)

    assert result_chunks[0].content == "<think>First thought"
    assert result_chunks[1].content == "Second thought"
    assert result_chunks[2].content == "</think>Final answer"


def test_message_format():
    strs = [
        "Hello",
        "Hello",
        "Hello",
    ]
    format_str = message_format(strs)
    assert format_str == "-Hello\n-Hello\n-Hello"

    documents = [
        Document(page_content="Hello"),
        Document(page_content="Hello"),
        Document(page_content="Hello"),
    ]
    formatted_message = message_format(documents)
    assert formatted_message == "-Hello\n-Hello\n-Hello"

    messages = [
        AIMessage(content="Hello"),
        AIMessage(content="Hello"),
        AIMessage(content="Hello"),
    ]
    formatted_message = message_format(messages)
    assert formatted_message == "-Hello\n-Hello\n-Hello"

    messages = [
        AIMessage(content="Hello"),
        AIMessage(content="Hello"),
        AIMessage(content="Hello"),
    ]
    formatted_message = message_format(messages, with_num=True)
    assert formatted_message == "-1. Hello\n-2. Hello\n-3. Hello"

    messages = [
        AIMessage(content="Hello"),
        AIMessage(content="Hello"),
        AIMessage(content="Hello"),
    ]
    formatted_message = message_format(messages, with_num=True, separator="|")
    assert formatted_message == "|1. Hello\n|2. Hello\n|3. Hello"


def test_has_tool_calling():
    message = AIMessage(
        content="Hello",
        tool_calls=[ToolCall(id="1", name="tool1", args={"arg1": "value1"})],
    )
    assert has_tool_calling(message)

    message = AIMessage(content="Hello")
    assert not has_tool_calling(message)

    message = AIMessage(content="Hello", tool_calls=[])
    assert not has_tool_calling(message)


def test_parse_tool_call():
    message = AIMessage(
        content="Hello",
        tool_calls=[
            ToolCall(id="1", name="tool1", args={"arg1": "value1"}),
            ToolCall(id="2", name="tool2", args={"arg2": "value2"}),
        ],
    )
    assert parse_tool_calling(message) == [
        ("tool1", {"arg1": "value1"}),
        ("tool2", {"arg2": "value2"}),
    ]
    assert parse_tool_calling(message, first_tool_call_only=True) == (
        "tool1",
        {"arg1": "value1"},
    )


def test_merge_ai_message_chunk():
    chunks = [
        AIMessageChunk(content="Chunk 1"),
        AIMessageChunk(content="Chunk 2"),
    ]
    merged_message = merge_ai_message_chunk(chunks)
    assert merged_message.content == "Chunk 1Chunk 2"
