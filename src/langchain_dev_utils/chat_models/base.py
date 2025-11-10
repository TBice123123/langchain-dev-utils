import os
from typing import Any, Literal, NotRequired, Optional, TypedDict, Union, cast

from langchain.chat_models.base import _SUPPORTED_PROVIDERS, _init_chat_model_helper
from langchain_core.language_models.chat_models import BaseChatModel

_MODEL_PROVIDERS_DICT = {}


ChatModelType = Union[type[BaseChatModel], Literal["openai-compatible"]]


ToolChoiceType = list[Literal["auto", "none", "any", "required", "specific"]]


class ChatModelProvider(TypedDict):
    provider: str
    chat_model: ChatModelType
    base_url: NotRequired[str]
    tool_choice: NotRequired[ToolChoiceType]
    keep_reasoning_content: NotRequired[bool]


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
    **kwargs: Any,
) -> BaseChatModel:
    """Helper function to load chat model.

    Args:
        model: Model name
        model_provider: Optional provider name
        **kwargs: Additional arguments for model initialization

    Returns:
        BaseChatModel: Initialized chat model instance
    """
    model, model_provider = _parse_model(model, model_provider)
    if model_provider in _MODEL_PROVIDERS_DICT.keys():
        chat_model = _MODEL_PROVIDERS_DICT[model_provider]["chat_model"]
        return chat_model(model=model, **kwargs)

    return _init_chat_model_helper(model, model_provider=model_provider, **kwargs)


def register_model_provider(
    provider_name: str,
    chat_model: ChatModelType,
    base_url: Optional[str] = None,
    tool_choice: Optional[ToolChoiceType] = None,
    keep_reasoning_content: bool = False,
):
    """Register a new model provider.

    This function allows you to register custom chat model providers that can be used
    with the load_chat_model function. It supports both custom model classes and
    string identifiers for supported providers.

    Args:
        provider_name: Name of the provider to register
        chat_model: Either a BaseChatModel class or a string identifier for a supported provider
        base_url: Optional base URL for API endpoints (Optional parameter; effective only when `chat_model` is a string.)
        tool_choice: Optional tool choice for the model.
            It is a list of unique values representing **auto**, **none**, **any**, **required**, and **specific**. (Optional parameter; effective only when `chat_model` is a string.)
        keep_reasoning_content: whether to retain the model's reasoning content in subsequent messages, defaults to `False`, only for reasoning models (optional parameter;effective only when `chat_model` is a string.)

    Raises:
        ValueError: If base_url is not provided when chat_model is a string,
                   or if chat_model string is not in supported providers

    Example:
        Basic usage with custom model class:
        >>> from langchain_dev_utils.chat_models import register_model_provider, load_chat_model
        >>> from langchain_core.language_models.fake_chat_models import FakeChatModel
        >>>
        >>> # Register custom model provider
        >>> register_model_provider("fakechat", FakeChatModel)
        >>> model = load_chat_model(model="fakechat:fake-model")
        >>> model.invoke("Hello")
        >>>
        >>> # Using with OpenAI-compatible API:
        >>> register_model_provider("vllm","openai-compatible",base_url="http://localhost:8000/v1")
        >>> model = load_chat_model(model="vllm:qwen3-4b")
        >>> model.invoke("Hello")
    """
    if isinstance(chat_model, str):
        try:
            from .adapters.openai_compatible import _create_openai_compatible_model
        except ImportError:
            raise ImportError(
                "Please install langchain_dev_utils[standard],when chat_model is a 'openai-compatible'"
            )

        base_url = base_url or os.getenv(f"{provider_name.upper()}_API_BASE")
        if base_url is None:
            raise ValueError(
                f"base_url must be provided or set {provider_name.upper()}_API_BASE environment variable when chat_model is a string"
            )

        if chat_model != "openai-compatible":
            raise ValueError(
                "when chat_model is a string, the value must be 'openai-compatible'"
            )
        chat_model = _create_openai_compatible_model(
            provider_name, base_url, tool_choice, keep_reasoning_content
        )
        _MODEL_PROVIDERS_DICT.update({provider_name: {"chat_model": chat_model}})
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
            - base_url: Optional base URL for API endpoints (Optional parameter; effective only when `chat_model` is a string.)
            - tool_choice: Optional tool choice for the model.
                It is a list of unique values representing **auto**, **none**, **any**, **required**, and **specific**. (Optional parameter; effective only when `chat_model` is a string.)
            - keep_reasoning_content: whether to retain the model's reasoning content in subsequent messages, defaults to `False`, only for reasoning models (optional parameter;effective only when `chat_model` is a string.)

    Raises:
        ValueError: If any of the providers are invalid

    Example:
        Register multiple providers at once:
        >>> from langchain_dev_utils.chat_models import batch_register_model_provider, load_chat_model
        >>> from langchain_core.language_models.fake_chat_models import FakeChatModel
        >>>
        >>> batch_register_model_provider([
        ...     {
        ...         "provider": "fakechat",
        ...         "chat_model": FakeChatModel,
        ...     },
        ...     {
        ...         "provider": "vllm",
        ...         "chat_model": "openai-compatible",
        ...         "base_url": "http://localhost:8000/v1",
        ...     },
        ... ])
        >>> model = load_chat_model(model="fakechat:fake-model")
        >>> model.invoke("Hello")
        >>> model = load_chat_model(model="vllm:qwen3-4b")
        >>> model.invoke("Hello")
    """

    for provider in providers:
        register_model_provider(
            provider["provider"],
            provider["chat_model"],
            provider.get("base_url"),
            tool_choice=provider.get("tool_choice"),
            keep_reasoning_content=provider.get("keep_reasoning_content", False),
        )


def load_chat_model(
    model: str,
    *,
    model_provider: Optional[str] = None,
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
        **kwargs: Additional arguments for model initialization (e.g., temperature, api_key)

    Returns:
        BaseChatModel: Initialized chat model instance

    Example:
        Load model with provider prefix:
        >>> from langchain_dev_utils.chat_models import load_chat_model
        >>> model = load_chat_model("vllm:qwen3-4b")
        >>> model.invoke("hello")

        Load model with separate provider parameter:
        >>> model = load_chat_model("qwen3-4b", model_provider="vllm")
        >>> model.invoke("hello")

        Load model with additional parameters:
        >>> model = load_chat_model(
        ...     "vllm:qwen3-4b",
        ...     temperature=0.7
        ... )
        >>> model.invoke("Hello, how are you?")
    """
    return _load_chat_model_helper(
        cast(str, model),
        model_provider=model_provider,
        **kwargs,
    )
