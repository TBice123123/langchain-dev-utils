from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
import pytest

from langchain_dev_utils.embeddings import (
    batch_register_embeddings_provider,
    load_embeddings,
)

load_dotenv()


batch_register_embeddings_provider(
    [
        {
            "provider_name": "siliconflow",
            "embeddings_model": "openai-compatible",
        },
        {"provider_name": "dashscope", "embeddings_model": DashScopeEmbeddings},
    ]
)


@pytest.fixture(params=["dashscope:text-embedding-v4", "siliconflow:BAAI/bge-m3"])
def embbeding_model(request: pytest.FixtureRequest) -> Embeddings:
    params = request.param
    return load_embeddings(params)


def test_embbedings(
    embbeding_model: Embeddings,
):
    assert embbeding_model.embed_query("what's your name")


@pytest.mark.asyncio
async def test_embbedings_async(
    embbeding_model: Embeddings,
):
    assert await embbeding_model.aembed_query("what's your name")
