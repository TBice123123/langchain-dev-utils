from typing import Any, cast

from langchain_core.embeddings.embeddings import Embeddings
from langchain_tests.integration_tests.embeddings import EmbeddingsIntegrationTests

from langchain_dev_utils.embeddings.adapters import create_openai_compatible_embedding

ZAIEmbeddings = create_openai_compatible_embedding(
    "zai", embedding_model_cls_name="ZAIEmbeddings"
)


class TestStandard(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> type[Embeddings]:
        """Embeddings class."""
        return cast("type[Embeddings]", ZAIEmbeddings)

    @property
    def embedding_model_params(self) -> dict[str, Any]:
        """Embeddings model parameters."""
        return {"model": "embedding-3"}
