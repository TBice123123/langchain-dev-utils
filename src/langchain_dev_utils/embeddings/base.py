from typing import Any, Literal, NotRequired, Optional, TypedDict, Union

from langchain.embeddings.base import _SUPPORTED_PROVIDERS, Embeddings, init_embeddings
from langchain_core.utils import from_env, secret_from_env

from langchain_dev_utils._utils import (
    _check_langchain_openai_install,
    _get_base_url_field_name,
)

_EMBEDDINGS_PROVIDERS_DICT = {}

EmbeddingsType = Union[type[Embeddings], Literal["openai-compatible"]]


class EmbeddingProvider(TypedDict):
    provider_name: str
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
        embeddings_model: Either an Embeddings class or a string identifier
            for a supported provider
        base_url: The API address of the Embedding model provider (optional,
            valid for both types of `embeddings_model`, but mainly used when
            `embeddings_model` is a string and is "openai-compatible")

    Raises:
        ValueError: If base_url is not provided when embeddings_model is a string

    Example:
        # Register with custom model class:
        >>> from langchain_dev_utils.embeddings import register_embeddings_provider, load_embeddings
        >>> from langchain_core.embeddings.fake import FakeEmbeddings
        >>>
        >>> register_embeddings_provider("fakeembeddings", FakeEmbeddings)
        >>> embeddings = load_embeddings("fakeembeddings:fake-embeddings",size=1024)
        >>> embeddings.embed_query("hello world")

        # Register with OpenAI-compatible API:
        >>> register_embeddings_provider(
        ...     "vllm",
        ...     "openai-compatible",
        ...     base_url="http://localhost:8000/v1"
        ... )
        >>> embeddings = load_embeddings("vllm:qwen3-embedding-4b")
        >>> embeddings.embed_query("hello world")
    """

    base_url = base_url or from_env(f"{provider_name.upper()}_API_BASE", default=None)()
    if isinstance(embeddings_model, str):
        if base_url is None:
            raise ValueError(
                f"base_url must be provided or set {provider_name.upper()}_API_BASE environment variable when embeddings_model is a string"
            )

        if embeddings_model != "openai-compatible":
            raise ValueError(
                "when embeddings_model is a string, the value must be 'openai-compatible'"
            )

        _check_langchain_openai_install()

        _EMBEDDINGS_PROVIDERS_DICT.update(
            {
                provider_name: {
                    "embeddings_model": embeddings_model,
                    "base_url": base_url,
                }
            }
        )
    else:
        if base_url is not None:
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

    This function allows you to register multiple embeddings providers at once,
    which is useful when setting up applications that need to work with multiple
    embedding services.

    Args:
        providers: List of EmbeddingProvider dictionaries, each containing:
            - provider_name: str - Provider name
            - embeddings_model: Union[Type[Embeddings], str] - Model class or
                provider string
            - base_url: The API address of the Embedding model provider
                (optional, valid for both types of `embeddings_model`, but
                mainly used when `embeddings_model` is a string and is
                "openai-compatible")

    Raises:
        ValueError: If any of the providers are invalid

    Example:
        # Register multiple providers at once:
        >>> from langchain_dev_utils.embeddings import batch_register_embeddings_provider, load_embeddings
        >>> from langchain_core.embeddings.fake import FakeEmbeddings
        >>>
        >>> batch_register_embeddings_provider(
        ...     [
        ...         {
        ...             "provider_name": "fakeembeddings",
        ...             "embeddings_model": FakeEmbeddings,
        ...         },
        ...         {
        ...             "provider_name": "vllm",
        ...             "embeddings_model": "openai-compatible",
        ...             "base_url": "http://localhost:8000/v1"
        ...         },
        ...     ]
        ... )
        >>> embeddings = load_embeddings("vllm:qwen3-embedding-4b")
        >>> embeddings.embed_query("hello world")
        >>> embeddings = load_embeddings("fakeembeddings:fake-embeddings", size=1024)
        >>> embeddings.embed_query("hello world")
    """
    for provider in providers:
        register_embeddings_provider(
            provider["provider_name"],
            provider["embeddings_model"],
            provider.get("base_url"),
        )


def load_embeddings(
    model: str,
    *,
    provider: Optional[str] = None,
    **kwargs: Any,
) -> Embeddings:
    """Load embeddings model.

    This function loads an embeddings model from the registered providers. The model parameter
    must be specified in the format "provider:model-name" when provider is not specified separately.

    Args:
        model: Model name in format 'provider:model-name' if provider not specified separately
        provider: Optional provider name (if not included in model parameter)
        **kwargs: Additional arguments for model initialization (e.g., api_key)

    Returns:
        Embeddings: Initialized embeddings model instance

    Raises:
        ValueError: If provider is not registered or API key is not found

    Example:
        # Load model with provider prefix:
        >>> from langchain_dev_utils.embeddings import load_embeddings
        >>> embeddings = load_embeddings("vllm:qwen3-embedding-4b")
        >>> embeddings.embed_query("hello world")

        # Load model with separate provider parameter:
        >>> embeddings = load_embeddings("qwen3-embedding-4b", provider="vllm")
        >>> embeddings.embed_query("hello world")
    """
    if provider is None:
        provider, model = _parse_model_string(model)
    if provider not in list(_EMBEDDINGS_PROVIDERS_DICT.keys()) + list(
        _SUPPORTED_PROVIDERS
    ):
        raise ValueError(f"Provider {provider} not registered")

    if provider in _SUPPORTED_PROVIDERS:
        return init_embeddings(model, provider=provider, **kwargs)

    embeddings = _EMBEDDINGS_PROVIDERS_DICT[provider]["embeddings_model"]
    if isinstance(embeddings, str):
        if not (api_key := kwargs.get("api_key")):
            api_key = secret_from_env(f"{provider.upper()}_API_KEY", default=None)()
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
        if base_url := _EMBEDDINGS_PROVIDERS_DICT[provider].get("base_url"):
            url_key = _get_base_url_field_name(embeddings)
            if url_key is not None:
                kwargs.update({url_key: base_url})
        return embeddings(model=model, **kwargs)
