import pytest
from langchain_core.embeddings import Embeddings

from langchain_dev_utils.embeddings import load_embeddings


@pytest.fixture(
    params=[
        "dashscope:text-embedding-v4",
        "ollama:bge-m3:latest",
    ]
)
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
