from functools import reduce
from typing import AsyncIterator, Iterator, Sequence, Tuple, cast

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessageChunk


def convert_reasoning_content_for_ai_message(
    model_response: AIMessage,
    think_tag: Tuple[str, str] = ("<think>", "</think>"),
) -> AIMessage:
    """Convert reasoning content in AI message to visible content.

    Args:
        model_response: AI message response from model
        think_tag: Tuple of (opening_tag, closing_tag) to wrap reasoning content

    Returns:
        AIMessage: Modified AI message with reasoning content in visible content
    """
    if "reasoning_content" in model_response.additional_kwargs:
        model_response.content = f"{think_tag[0]}{model_response.additional_kwargs['reasoning_content']}{think_tag[1]}{model_response.content}"
    return model_response


def convert_reasoning_content_for_chunk_iterator(
    model_response: Iterator[BaseMessageChunk],
    think_tag: Tuple[str, str] = ("<think>", "</think>"),
) -> Iterator[BaseMessageChunk]:
    """Convert reasoning content for streaming response chunks.

    Args:
        model_response: Iterator of message chunks from streaming response
        think_tag: Tuple of (opening_tag, closing_tag) to wrap reasoning content

    Yields:
        BaseMessageChunk: Modified message chunks with reasoning content
    """
    isfirst = True
    isend = True

    for chunk in model_response:
        if (
            isinstance(chunk, AIMessageChunk)
            and "reasoning_content" in chunk.additional_kwargs
        ):
            if isfirst:
                chunk.content = (
                    f"{think_tag[0]}{chunk.additional_kwargs['reasoning_content']}"
                )
                isfirst = False
            else:
                chunk.content = chunk.additional_kwargs["reasoning_content"]
        elif (
            isinstance(chunk, AIMessageChunk)
            and "reasoning_content" not in chunk.additional_kwargs
            and chunk.content
            and isend
        ):
            chunk.content = f"{think_tag[1]}{chunk.content}"
            isend = False
        yield chunk


async def aconvert_reasoning_content_for_chunk_iterator(
    model_response: AsyncIterator[BaseMessageChunk],
    think_tag: Tuple[str, str] = ("<think>", "</think>"),
) -> AsyncIterator[BaseMessageChunk]:
    """Async convert reasoning content for streaming response chunks.

    Args:
        model_response: Async iterator of message chunks from streaming response
        think_tag: Tuple of (opening_tag, closing_tag) to wrap reasoning content

    Yields:
        BaseMessageChunk: Modified message chunks with reasoning content
    """
    isfirst = True
    isend = True

    async for chunk in model_response:
        if (
            isinstance(chunk, AIMessageChunk)
            and "reasoning_content" in chunk.additional_kwargs
        ):
            if isfirst:
                chunk.content = (
                    f"{think_tag[0]}{chunk.additional_kwargs['reasoning_content']}"
                )
                isfirst = False
            else:
                chunk.content = chunk.additional_kwargs["reasoning_content"]
        elif (
            isinstance(chunk, AIMessageChunk)
            and "reasoning_content" not in chunk.additional_kwargs
            and chunk.content
            and isend
        ):
            chunk.content = f"{think_tag[1]}{chunk.content}"
            isend = False
        yield chunk


def merge_ai_message_chunk(chunks: Sequence[AIMessageChunk]) -> AIMessage:
    """Merge a sequence of AIMessageChunk into a single AIMessage.

    Args:
        chunks: Sequence of AIMessageChunk to merge

    Returns:
        AIMessage: Merged AIMessage
    """
    ai_message_chunk = cast(AIMessageChunk, reduce(lambda x, y: x + y, chunks))
    ai_message_chunk.additional_kwargs.pop("tool_calls", None)

    data = {
        "id": ai_message_chunk.id,
        "content": ai_message_chunk.content,
        "response_metadata": ai_message_chunk.response_metadata,
        "additional_kwargs": ai_message_chunk.additional_kwargs,
    }
    if hasattr(ai_message_chunk, "tool_calls") and len(ai_message_chunk.tool_calls):
        data["tool_calls"] = ai_message_chunk.tool_calls
    return AIMessage.model_validate(data)
