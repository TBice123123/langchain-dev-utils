from langchain_core.messages import AIMessageChunk
from langchain_dev_utils.content import merge_ai_message_chunk


def test_merge_ai_message_chunk():
    chunks = [
        AIMessageChunk(content="Chunk 1"),
        AIMessageChunk(content="Chunk 2"),
    ]
    merged_message = merge_ai_message_chunk(chunks)
    assert merged_message.content == "Chunk 1Chunk 2"
