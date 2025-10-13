from typing import cast

from dotenv import load_dotenv
from langchain.embeddings.base import Embeddings
from langchain_siliconflow.embeddings import SiliconFlowEmbeddings
import pytest

from langchain_dev_utils import load_embeddings, batch_register_embeddings_provider

load_dotenv()


batch_register_embeddings_provider(
    [
        {"provider": "dashscope", "embeddings_model": "openai-compatible"},
        {"provider": "siliconflow", "embeddings_model": SiliconFlowEmbeddings},
    ]
)


def test_embbedings():
    emb1 = cast(Embeddings, load_embeddings("dashscope:text-embedding-v4"))
    emb2 = cast(Embeddings, load_embeddings("siliconflow:BAAI/bge-m3"))

    assert emb1.embed_query("what's your name")
    assert emb2.embed_query("what's your name")


@pytest.mark.asyncio
async def test_embbedings_async():
    emb1 = cast(Embeddings, load_embeddings("dashscope:text-embedding-v4"))
    emb2 = cast(Embeddings, load_embeddings("siliconflow:BAAI/bge-m3"))

    assert await emb1.aembed_query("what's your name")
    assert await emb2.aembed_query("what's your name")
