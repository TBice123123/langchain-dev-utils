

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_dev_utils.content import (
    convert_reasoning_content_for_ai_message,
    convert_reasoning_content_for_chunk_iterator,
    aconvert_reasoning_content_for_ai_message,
    aconvert_reasoning_content_for_chunk_iterator,
)
import pytest


def test_convert_reasoning_content_for_ai_message():
    ai_message = AIMessage(
        content="Hello",
        additional_kwargs={"reasoning_content": "I think therefore I am"}
    )
    
    result = convert_reasoning_content_for_ai_message(ai_message, ("<think>", "</think>"))
    assert result.content == "<think>I think therefore I am</think>"
    
    result = convert_reasoning_content_for_ai_message(ai_message)
    assert result.content == "I think therefore I am"


def test_convert_reasoning_content_for_chunk_iterator():
    chunks = [
        AIMessageChunk(content="Hello", additional_kwargs={"reasoning_content": "First thought"}),
        AIMessageChunk(content="Hello", additional_kwargs={"reasoning_content": "Second thought"}),
        AIMessageChunk(content="Final answer"),
    ]
    
    result_chunks = list(convert_reasoning_content_for_chunk_iterator(iter(chunks), ("<think>", "</think>")))
    
    assert result_chunks[0].content == "<think>First thought"
    assert result_chunks[1].content == "Second thought"
    assert result_chunks[2].content == "</think>Final answer"


@pytest.mark.asyncio
async def test_aconvert_reasoning_content_for_ai_message():
    ai_message = AIMessage(
        content="Hello",
        additional_kwargs={"reasoning_content": "I think therefore I am"}
    )
    
    result = await aconvert_reasoning_content_for_ai_message(ai_message, ("<think>", "</think>"))
    assert result.content == "<think>I think therefore I am</think>"


@pytest.mark.asyncio
async def test_aconvert_reasoning_content_for_chunk_iterator():
    async def async_chunk_generator():
        chunks = [
            AIMessageChunk(content="Hello", additional_kwargs={"reasoning_content": "First thought"}),
            AIMessageChunk(content="Hello", additional_kwargs={"reasoning_content": "Second thought"}),
            AIMessageChunk(content="Final answer"),
        ]
        for chunk in chunks:
            yield chunk
    
    result_chunks = []
    async for chunk in aconvert_reasoning_content_for_chunk_iterator(async_chunk_generator(), ("<think>", "</think>")):
        result_chunks.append(chunk)
    
    assert result_chunks[0].content == "<think>First thought"
    assert result_chunks[1].content == "Second thought"
    assert result_chunks[2].content == "</think>Final answer"