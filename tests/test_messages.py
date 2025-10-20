from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
import pytest

from langchain_dev_utils.message_convert import (
    aconvert_reasoning_content_for_chunk_iterator,
    convert_reasoning_content_for_ai_message,
    convert_reasoning_content_for_chunk_iterator,
    format_sequence,
    merge_ai_message_chunk,
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


@pytest.mark.parametrize(
    "input_data,expected_output,with_num,separator",
    [
        # Test with list of strings
        (
            ["Hello", "Hello", "Hello"],
            "-Hello\n-Hello\n-Hello",
            False,
            "-",
        ),
        # Test with list of Document objects
        (
            [
                Document(page_content="Hello"),
                Document(page_content="Hello"),
                Document(page_content="Hello"),
            ],
            "-Hello\n-Hello\n-Hello",
            False,
            "-",
        ),
        # Test with list of AIMessage objects
        (
            [
                AIMessage(content="Hello"),
                AIMessage(content="Hello"),
                AIMessage(content="Hello"),
            ],
            "-Hello\n-Hello\n-Hello",
            False,
            "-",
        ),
        # Test with numbering
        (
            [
                AIMessage(content="Hello"),
                AIMessage(content="Hello"),
                AIMessage(content="Hello"),
            ],
            "-1. Hello\n-2. Hello\n-3. Hello",
            True,
            "-",
        ),
        # Test with custom separator and numbering
        (
            [
                AIMessage(content="Hello"),
                AIMessage(content="Hello"),
                AIMessage(content="Hello"),
            ],
            "|1. Hello\n|2. Hello\n|3. Hello",
            True,
            "|",
        ),
    ],
)
def test_message_format(
    input_data: list[Document] | list[str] | list[BaseMessage],
    expected_output: str,
    with_num: bool,
    separator: str,
):
    if with_num:
        if separator != "-":
            formatted_message = format_sequence(
                input_data, with_num=True, separator=separator
            )
        else:
            formatted_message = format_sequence(input_data, with_num=True)
    else:
        formatted_message = format_sequence(input_data)

    assert formatted_message == expected_output


def test_merge_ai_message_chunk():
    chunks = [
        AIMessageChunk(content="Chunk 1"),
        AIMessageChunk(content="Chunk 2"),
    ]
    merged_message = merge_ai_message_chunk(chunks)
    assert merged_message.content == "Chunk 1Chunk 2"
