from typing import Optional, cast

from langchain_core.utils import from_env

from langchain_dev_utils._utils import _check_pkg_install


def create_openai_compatible_embedding(
    embedding_provider: str,
    base_url: Optional[str] = None,
    embedding_model_cls_name: Optional[str] = None,
):
    """Factory function for creating provider-specific OpenAI-compatible embedding classes.

    Dynamically generates embedding classes for different OpenAI-compatible providers,
    configuring environment variable mappings and default base URLs specific to each provider.

    Args:
        embedding_provider (str): Identifier for the OpenAI-compatible provider (e.g. `vllm`, `moonshot`)
        base_url (Optional[str], optional): Default API base URL for the provider. Defaults to None. If not provided, will try to use the environment variable.
        embedding_model_cls_name (Optional[str], optional): Optional custom class name for the generated embedding. Defaults to None.
    Returns:
        Type[_BaseEmbeddingOpenAICompatible]: Configured embedding class ready for instantiation with provider-specific settings

    Examples:
        >>> from langchain_dev_utils.embeddings.adapters import create_openai_compatible_embedding
        >>> VLLMEmbedding = create_openai_compatible_embedding(
        ...     "vllm",
        ...     base_url="http://localhost:8000",
        ...     embedding_model_cls_name="VLLMEmbedding",
        ... )
        >>> model = VLLMEmbedding(model="qwen3-embedding-8b")
        >>> model.embed_query("hello")
    """
    _check_pkg_install("langchain_openai")
    from .openai_compatible import _create_openai_compatible_embedding

    base_url = (
        base_url or from_env(f"{embedding_provider.upper()}_API_BASE", default=None)()
    )
    return _create_openai_compatible_embedding(
        provider=embedding_provider,
        base_url=cast(str, base_url),
        embeddings_cls_name=embedding_model_cls_name,
    )
