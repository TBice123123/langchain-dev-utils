import os
from typing import Any, NotRequired, Optional, TypedDict, Union, cast

from langchain.chat_models.base import (
    _SUPPORTED_PROVIDERS,
    BaseChatModel,
    _init_chat_model_helper,
    init_chat_model,
)

_MODEL_PROVIDERS_DICT = {}


class ChatModelProvider(TypedDict):
    provider: str
    chat_model: Union[type[BaseChatModel], str]
    base_url: NotRequired[str]


def _parse_model(model: str, model_provider: Optional[str]) -> tuple[str, str]:
    """Parse model string and provider.

    Args:
        model: Model name string, potentially including provider prefix
        model_provider: Optional provider name

    Returns:
        Tuple of (model_name, provider_name)

    Raises:
        ValueError: If unable to infer model provider
    """
    support_providers = list(_MODEL_PROVIDERS_DICT.keys()) + list(_SUPPORTED_PROVIDERS)
    if not model_provider and ":" in model and model.split(":")[0] in support_providers:
        model_provider = model.split(":")[0]
        model = ":".join(model.split(":")[1:])
    if not model_provider:
        msg = (
            f"Unable to infer model provider for {model=}, please specify "
            f"model_provider directly."
        )
        raise ValueError(msg)
    model_provider = model_provider.replace("-", "_").lower()
    return model, model_provider


def _load_chat_model_helper(
    model: str,
    model_provider: Optional[str] = None,
    enable_reasoning_parse: bool = False,
    **kwargs: Any,
) -> BaseChatModel:
    """Helper function to load chat model.

    Args:
        model: Model name
        model_provider: Optional provider name
        enable_reasoning_parse: Optional enable reasoning parse
        **kwargs: Additional arguments for model initialization

    Returns:
        BaseChatModel: Initialized chat model instance
    """
    model, model_provider = _parse_model(model, model_provider)
    if model_provider in _MODEL_PROVIDERS_DICT.keys():
        chat_model = _MODEL_PROVIDERS_DICT[model_provider]["chat_model"]
        if isinstance(chat_model, str):
            if not (api_key := kwargs.get("api_key")):
                api_key = os.getenv(f"{model_provider.upper()}_API_KEY")
                if not api_key:
                    raise ValueError(
                        f"API key for {model_provider} not found. Please set it in the environment."
                    )
                kwargs["api_key"] = api_key
            base_url = _MODEL_PROVIDERS_DICT[model_provider]["base_url"]
            if chat_model == "openai" and enable_reasoning_parse:
                chat_model = "deepseek"

            if chat_model in ["openai", "anthropic"]:
                return init_chat_model(
                    model=model,
                    model_provider=chat_model,
                    base_url=base_url,
                    **kwargs,
                )
            else:
                return init_chat_model(
                    model=model,
                    model_provider=chat_model,
                    api_base=base_url,
                    **kwargs,
                )
        else:
            return chat_model(model=model, **kwargs)

    return _init_chat_model_helper(model, model_provider=model_provider, **kwargs)


def register_model_provider(
    provider_name: str,
    chat_model: Union[type[BaseChatModel], str],
    base_url: Optional[str] = None,
):
    """Register a new model provider.

    This function allows you to register custom chat model providers that can be used
    with the load_chat_model function. It supports both custom model classes and
    string identifiers for supported providers.

    Args:
        provider_name: Name of the provider to register
        chat_model: Either a BaseChatModel class or a string identifier for a supported provider
        base_url: Optional base URL for API endpoints (required when chat_model is a string)

    Raises:
        ValueError: If base_url is not provided when chat_model is a string,
                   or if chat_model string is not in supported providers

    Example:
        Basic usage with custom model class:
        >>> from langchain_dev_utils import register_model_provider, load_chat_model
        >>> from langchain_qwq import ChatQwen
        >>> from dotenv import load_dotenv
        >>>
        >>> load_dotenv()
        >>>
        >>> # Register custom model provider
        >>> register_model_provider("dashscope", ChatQwen)
        >>> model = load_chat_model(model="dashscope:qwen-flash")
        >>> model.invoke("Hello")

        Using with OpenAI-compatible API:
        >>> register_model_provider("openrouter", "openai", base_url="https://openrouter.ai/api/v1")
        >>> model = load_chat_model(model="openrouter:moonshotai/kimi-k2-0905")
        >>> model.invoke("Hello")
    """
    if isinstance(chat_model, str):
        base_url = base_url or os.getenv(f"{provider_name.upper()}_API_BASE")
        if base_url is None:
            raise ValueError(
                f"base_url must be provided or set {provider_name.upper()}_API_BASE environment variable when chat_model is a string"
            )

        if chat_model not in _SUPPORTED_PROVIDERS:
            raise ValueError(
                f"when chat_model is a string, the value must be one of {_SUPPORTED_PROVIDERS}"
            )

        _MODEL_PROVIDERS_DICT.update(
            {provider_name: {"chat_model": chat_model, "base_url": base_url}}
        )
    else:
        _MODEL_PROVIDERS_DICT.update({provider_name: {"chat_model": chat_model}})


def batch_register_model_provider(
    providers: list[ChatModelProvider],
):
    """Batch register model providers.

    This function allows you to register multiple model providers at once, which is
    useful when setting up applications that need to work with multiple model services.

    Args:
        providers: List of ChatModelProvider dictionaries, each containing:
            - provider: str - Provider name
            - chat_model: Union[Type[BaseChatModel], str] - Model class or provider string
            - base_url: Optional[str] - Base URL for API endpoints

    Raises:
        ValueError: If any of the providers are invalid

    Example:
        Register multiple providers at once:
        >>> from langchain_dev_utils import batch_register_model_provider, load_chat_model
        >>> from langchain_qwq import ChatQwen
        >>>
        >>> batch_register_model_provider([
        ...     {
        ...         "provider": "dashscope",
        ...         "chat_model": ChatQwen,
        ...     },
        ...     {
        ...         "provider": "openrouter",
        ...         "chat_model": "openai",
        ...         "base_url": "https://openrouter.ai/api/v1",
        ...     },
        ... ])
        >>> model = load_chat_model(model="dashscope:qwen-flash")
        >>> model.invoke("Hello")
        >>> model = load_chat_model(model="openrouter:moonshotai/kimi-k2-0905")
        >>> model.invoke("Hello")
    """

    for provider in providers:
        register_model_provider(
            provider["provider"], provider["chat_model"], provider.get("base_url")
        )


def load_chat_model(
    model: str,
    *,
    model_provider: Optional[str] = None,
    enable_reasoning_parse: bool = False,
    **kwargs: Any,
) -> BaseChatModel:
    """Load a chat model.

    This function loads a chat model from the registered providers. The model parameter
    can be specified in two ways:
    1. "provider:model-name" - When model_provider is not specified
    2. "model-name" - When model_provider is specified separately

    Args:
        model: Model name, either as "provider:model-name" or just "model-name"
        model_provider: Optional provider name (if not included in model parameter)
        enable_reasoning_parse: optional parameter to enable reasoning parse when chat_model is 'openai', default is False
        **kwargs: Additional arguments for model initialization (e.g., temperature, api_key)

    Returns:
        BaseChatModel: Initialized chat model instance

    Example:
        Load model with provider prefix:
        >>> from langchain_dev_utils import load_chat_model
        >>> model = load_chat_model("dashscope:qwen3-235b-a22b-instruct-2507")
        >>> model.invoke("hello")

        Load model with separate provider parameter:
        >>> model = load_chat_model("qwen-flash", model_provider="dashscope")
        >>> model.invoke("hello")

        Load model with additional parameters:
        >>> model = load_chat_model(
        ...     "dashscope:qwen-flash",
        ...     temperature=0.7,
        ...     enable_thinking=True
        ... )
        >>> model.invoke("Hello, how are you?")
    """
    return _load_chat_model_helper(
        cast(str, model),
        model_provider=model_provider,
        enable_reasoning_parse=enable_reasoning_parse,
        **kwargs,
    )
