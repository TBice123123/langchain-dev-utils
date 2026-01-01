from typing import Any, cast

from langchain_core.embeddings.embeddings import Embeddings
from langchain_tests.integration_tests.embeddings import EmbeddingsIntegrationTests

from langchain_dev_utils.embeddings.adapters import create_openai_compatible_embedding

SiliconFlowEmbeddings = create_openai_compatible_embedding(
    "siliconflow", embedding_model_cls_name="SiliconFlowEmbeddings"
)


class TestStandard(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> type[Embeddings]:
        """Embeddings class."""
        return cast("type[Embeddings]", SiliconFlowEmbeddings)

    @property
    def embedding_model_params(self) -> dict[str, Any]:
        """Embeddings model parameters."""
        return {"model": "BAAI/bge-m3"}
