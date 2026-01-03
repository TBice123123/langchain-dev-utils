from typing import Optional, Type

from langchain_core.utils import from_env, secret_from_env
from langchain_openai.embeddings import OpenAIEmbeddings
from pydantic import Field, SecretStr, create_model

from ..._utils import _validate_base_url, _validate_model_cls_name


class _BaseEmbeddingOpenAICompatible(OpenAIEmbeddings):
    """Base class for OpenAI-Compatible embeddings.

    This class extends the OpenAIEmbeddings class to support
    custom API keys and base URLs for OpenAI-Compatible models.

    Note: This is a template class and should not be exported or instantiated
    directly. Instead, use it as a base class and provide the specific provider
    name through inheritance or the factory function
    `create_openai_compatible_embedding()`.
    """

    openai_api_key: Optional[SecretStr] = Field(
        default_factory=secret_from_env("OPENAI_COMPATIBLE_API_KEY", default=None),
        alias="api_key",
    )
    """OpenAI Compatible API key"""
    openai_api_base: str = Field(
        default_factory=from_env("OPENAI_COMPATIBLE_API_BASE", default=""),
        alias="base_url",
    )
    """OpenAI Compatible API base URL"""

    check_embedding_ctx_length: bool = False
    """Whether to check the token length of inputs and automatically split inputs
        longer than embedding_ctx_length. Defaults to False. """


def _create_openai_compatible_embedding(
    provider: str,
    base_url: str,
    embeddings_cls_name: Optional[str] = None,
) -> Type[_BaseEmbeddingOpenAICompatible]:
    """Factory function for creating provider-specific OpenAI-compatible embeddings classes.

    Dynamically generates embeddings classes for different OpenAI-compatible providers,
    configuring environment variable mappings and default base URLs specific to each provider.

    Args:
        provider: Provider identifier (e.g.`vllm`)
        base_url: Default API base URL for the provider
        embeddings_cls_name: Optional custom class name for the generated embeddings. Defaults to None.

    Returns:
        Configured embeddings class ready for instantiation with provider-specific settings
    """
    embeddings_cls_name = embeddings_cls_name or f"{provider.title()}Embeddings"

    if len(provider) >= 20:
        raise ValueError(
            f"provider must be less than 50 characters. Received: {provider}"
        )

    _validate_model_cls_name(embeddings_cls_name)

    _validate_base_url(base_url)

    return create_model(
        embeddings_cls_name,
        __base__=_BaseEmbeddingOpenAICompatible,
        openai_api_base=(
            str,
            Field(
                default_factory=from_env(
                    f"{provider.upper()}_API_BASE", default=base_url
                ),
            ),
        ),
        openai_api_key=(
            str,
            Field(
                default_factory=secret_from_env(
                    f"{provider.upper()}_API_KEY", default=None
                ),
            ),
        ),
    )
