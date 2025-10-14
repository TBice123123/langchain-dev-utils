import os
from typing import Any, Literal, Optional, Union

from langchain.embeddings.base import Embeddings, _SUPPORTED_PROVIDERS, init_embeddings
from langchain_core.runnables import Runnable
from typing import TypedDict, NotRequired


_EMBEDDINGS_PROVIDERS_DICT = {}

EmbeddingsType = Union[type[Embeddings], Literal["openai-compatible"]]


class EmbeddingProvider(TypedDict):
    provider: str
    embeddings_model: EmbeddingsType
    base_url: NotRequired[str]


def _parse_model_string(model_name: str) -> tuple[str, str]:
    """Parse model string into provider and model name.

    Args:
        model_name: Model name string in format 'provider:model-name'

    Returns:
        Tuple of (provider, model) parsed from the model_name

    Raises:
        ValueError: If model name format is invalid or model name is empty
    """
    if ":" not in model_name:
        msg = (
            f"Invalid model format '{model_name}'.\n"
            f"Model name must be in format 'provider:model-name'\n"
        )
        raise ValueError(msg)

    provider, model = model_name.split(":", 1)
    provider = provider.lower().strip()
    model = model.strip()
    if not model:
        msg = "Model name cannot be empty"
        raise ValueError(msg)
    return provider, model


def register_embeddings_provider(
    provider_name: str,
    embeddings_model: EmbeddingsType,
    base_url: Optional[str] = None,
):
    """Register an embeddings provider.

    This function allows you to register custom embeddings providers that can be used
    with the load_embeddings function. It supports both custom model classes and
    string identifiers for supported providers.

    Args:
        provider_name: Name of the provider to register
        embeddings_model: Either an Embeddings class or a string identifier for a supported provider
        base_url: Optional base URL for API endpoints (required when embeddings_model is a string)

    Raises:
        ValueError: If base_url is not provided when embeddings_model is a string

    Example:
        Register with custom model class:
        >>> from langchain_dev_utils import register_embeddings_provider, load_embeddings
        >>> from langchain_siliconflow import SiliconFlowEmbeddings
        >>>
        >>> register_embeddings_provider("siliconflow", SiliconFlowEmbeddings)
        >>> embeddings = load_embeddings("siliconflow:text-embedding-v4")
        >>> embeddings.embed_query("hello world")

        Register with OpenAI-compatible API:
        >>> register_embeddings_provider(
        ...     "dashscope", "openai-compatible", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        ... )
        >>> embeddings = load_embeddings("dashscope:text-embedding-v4")
        >>> embeddings.embed_query("hello world")
    """
    if isinstance(embeddings_model, str):
        base_url = base_url or os.getenv(f"{provider_name.upper()}_API_BASE")
        if base_url is None:
            raise ValueError(
                f"base_url must be provided or set {provider_name.upper()}_API_BASE environment variable when embeddings_model is a string"
            )

        if embeddings_model != "openai-compatible":
            raise ValueError(
                "when embeddings_model is a string, the value must be 'openai-compatible'"
            )

        _EMBEDDINGS_PROVIDERS_DICT.update(
            {
                provider_name: {
                    "embeddings_model": embeddings_model,
                    "base_url": base_url,
                }
            }
        )
    else:
        _EMBEDDINGS_PROVIDERS_DICT.update(
            {provider_name: {"embeddings_model": embeddings_model}}
        )


def batch_register_embeddings_provider(
    providers: list[EmbeddingProvider],
):
    """Batch register embeddings providers.

    This function allows you to register multiple embeddings providers at once, which is
    useful when setting up applications that need to work with multiple embedding services.

    Args:
        providers: List of EmbeddingProvider dictionaries, each containing:
            - provider: str - Provider name
            - embeddings_model: Union[Type[Embeddings], str] - Model class or provider string
            - base_url: Optional[str] - Base URL for API endpoints

    Raises:
        ValueError: If any of the providers are invalid

    Example:
        Register multiple providers at once:
        >>> from langchain_dev_utils import batch_register_embeddings_provider, load_embeddings
        >>> from langchain_siliconflow import SiliconFlowEmbeddings
        >>>
        >>> batch_register_embeddings_provider(
        ...     [
        ...         {"provider": "dashscope", "embeddings_model": "openai-compatible", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"},
        ...         {"provider": "siliconflow", "embeddings_model": SiliconFlowEmbeddings},
        ...     ]
        ... )
        >>> embeddings = load_embeddings("dashscope:text-embedding-v4")
        >>> embeddings.embed_query("hello world")
        >>> embeddings = load_embeddings("siliconflow:text-embedding-v4")
        >>> embeddings.embed_query("hello world")
    """
    for provider in providers:
        register_embeddings_provider(
            provider["provider"], provider["embeddings_model"], provider.get("base_url")
        )


def load_embeddings(
    model: str,
    *,
    provider: Optional[str] = None,
    **kwargs: Any,
) -> Union[Embeddings, Runnable[Any, list[float]]]:
    """Load embeddings model.

    This function loads an embeddings model from the registered providers. The model parameter
    must be specified in the format "provider:model-name" when provider is not specified separately.

    Args:
        model: Model name in format 'provider:model-name' if provider not specified separately
        provider: Optional provider name (if not included in model parameter)
        **kwargs: Additional arguments for model initialization (e.g., api_key)

    Returns:
        Union[Embeddings, Runnable[Any, list[float]]]: Initialized embeddings model instance

    Raises:
        ValueError: If provider is not registered or API key is not found

    Example:
        Load model with provider prefix:
        >>> from langchain_dev_utils import load_embeddings
        >>> embeddings = load_embeddings("dashscope:text-embedding-v4")
        >>> embeddings.embed_query("hello world")

        Load model with separate provider parameter:
        >>> embeddings = load_embeddings("text-embedding-v4", provider="dashscope")
        >>> embeddings.embed_query("hello world")

        Load model with additional parameters:
        >>> embeddings = load_embeddings(
        ...     "dashscope:text-embedding-v4",
        ...     api_key="your-api-key"
        ... )
        >>> embeddings.embed_query("hello world")

        Batch processing:
        >>> texts = ["hello world", "how are you", "good morning"]
        >>> embeddings_list = embeddings.embed_documents(texts)
        >>> len(embeddings_list)
        >>> len(embeddings_list[0])
    """
    if provider is None:
        provider, model = _parse_model_string(model)
    if provider not in list(_EMBEDDINGS_PROVIDERS_DICT.keys()) + list(
        _SUPPORTED_PROVIDERS
    ):
        raise ValueError(f"Provider {provider} not registered")

    embeddings = _EMBEDDINGS_PROVIDERS_DICT[provider]["embeddings_model"]
    if isinstance(embeddings, str):
        if not (api_key := kwargs.get("api_key")):
            api_key = os.getenv(f"{provider.upper()}_API_KEY")
            if not api_key:
                raise ValueError(
                    f"API key for {provider} not found. Please set it in the environment."
                )
            kwargs["api_key"] = api_key
            if embeddings == "openai-compatible":
                kwargs["check_embedding_ctx_length"] = False
                embeddings = "openai"

        return init_embeddings(
            model=model,
            provider=embeddings,
            base_url=_EMBEDDINGS_PROVIDERS_DICT[provider]["base_url"],
            **kwargs,
        )
    else:
        return embeddings(model=model, **kwargs)
