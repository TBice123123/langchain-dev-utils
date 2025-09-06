from dotenv import load_dotenv
from langchain.embeddings.base import Embeddings
from typing import cast
from langchain_dev_utils import load_embeddings, register_embeddings_provider
import pytest

load_dotenv()

register_embeddings_provider(
    "dashscope", "openai", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


def test_embbedings():
    emb = cast(Embeddings, load_embeddings("dashscope:text-embedding-v4"))

    assert emb.embed_query("what's your name")


@pytest.mark.asyncio
async def test_embbedings_async():
    emb = cast(Embeddings, load_embeddings("dashscope:text-embedding-v4"))

    assert await emb.aembed_query("what's your name")
