from typing import Any, NotRequired, Optional, TypedDict, cast

from langchain.chat_models.base import _SUPPORTED_PROVIDERS, _init_chat_model_helper
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.utils import from_env

from langchain_dev_utils._utils import (
    _check_langchain_openai_install,
    _get_base_url_field_name,
)

from .types import ChatModelType, CompatibilityOptions

_MODEL_PROVIDERS_DICT = {}


class ChatModelProvider(TypedDict):
    provider_name: str
    chat_model: ChatModelType
    base_url: NotRequired[str]
    model_profiles: NotRequired[dict[str, dict[str, Any]]]
    compatibility_options: NotRequired[CompatibilityOptions]


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
    if model_provider in _MODEL_PROVIDERS_DICT:
        chat_model = _MODEL_PROVIDERS_DICT[model_provider]["chat_model"]
        if base_url := _MODEL_PROVIDERS_DICT[model_provider].get("base_url"):
            url_key = _get_base_url_field_name(chat_model)
            if url_key:
                kwargs.update({url_key: base_url})
        if model_profiles := _MODEL_PROVIDERS_DICT[model_provider].get(
            "model_profiles"
        ):
            if model in model_profiles and "profile" not in kwargs:
                kwargs.update({"profile": model_profiles[model]})
        return chat_model(model=model, **kwargs)

    return _init_chat_model_helper(model, model_provider=model_provider, **kwargs)


def register_model_provider(
    provider_name: str,
    chat_model: ChatModelType,
    base_url: Optional[str] = None,
    model_profiles: Optional[dict[str, dict[str, Any]]] = None,
    compatibility_options: Optional[CompatibilityOptions] = None,
):
    """Register a new model provider.

    This function allows you to register custom chat model providers that can be used
    with the load_chat_model function. It supports both custom model classes and
    string identifiers for supported providers.

    Args:
        provider_name: The name of the model provider, used as an identifier for 
            loading models later.
        chat_model: The chat model, which can be either a `ChatModel` instance or 
            a string (currently only `"openai-compatible"` is supported).
        base_url: The API endpoint URL of the model provider (optional; applicable 
            to both `chat_model` types, but primarily used when `chat_model` is a 
            string with value `"openai-compatible"`).
        model_profiles: Declares the capabilities and parameters supported by each 
            model provided by this provider (optional; applicable to both `chat_model` 
            types). The configuration corresponding to the `model_name` will be loaded 
            and assigned to `model.profile` (e.g., fields such as `max_input_tokens`, 
            `tool_calling`etc.).
        compatibility_options: Compatibility options for the model provider (optional; 
            only effective when `chat_model` is a string with value `"openai-compatible"`). 
            Used to declare support for OpenAI-compatible features (e.g., `tool_choice` 
            strategies, JSON mode, etc.) to ensure correct functional adaptation.
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
    base_url = base_url or from_env(f"{provider_name.upper()}_API_BASE", default=None)()
    if isinstance(chat_model, str):
        _check_langchain_openai_install()
        from .adapters.openai_compatible import _create_openai_compatible_model

        if base_url is None:
            raise ValueError(
                f"base_url must be provided or set {provider_name.upper()}_API_BASE environment variable when chat_model is a string"
            )

        if chat_model != "openai-compatible":
            raise ValueError(
                "when chat_model is a string, the value must be 'openai-compatible'"
            )
        chat_model = _create_openai_compatible_model(
            provider_name,
            base_url,
            compatibility_options=compatibility_options,
        )
        _MODEL_PROVIDERS_DICT.update(
            {
                provider_name: {
                    "chat_model": chat_model,
                    "base_url": base_url,
                    "model_profiles": model_profiles,
                }
            }
        )
    else:
        if base_url is not None:
            _MODEL_PROVIDERS_DICT.update(
                {
                    provider_name: {
                        "chat_model": chat_model,
                        "base_url": base_url,
                        "model_profiles": model_profiles,
                    }
                }
            )
        else:
            _MODEL_PROVIDERS_DICT.update(
                {
                    provider_name: {
                        "chat_model": chat_model,
                        "model_profiles": model_profiles,
                    }
                }
            )


def batch_register_model_provider(
    providers: list[ChatModelProvider],
):
    """Batch register model providers.

    This function allows you to register multiple model providers at once, which is
    useful when setting up applications that need to work with multiple model services.

    Args:
        providers: List of ChatModelProvider dictionaries, each containing:
            - provider_name (str): The name of the model provider, used as an
              identifier for loading models later.
            - chat_model (ChatModel | str): The chat model, which can be either
              a `ChatModel` instance or a string (currently only `"openai-compatible"`
              is supported).
            - base_url (str, optional): The API endpoint URL of the model provider.
              Applicable to both `chat_model` types, but primarily used when `chat_model`
              is `"openai-compatible"`.
            - model_profiles (dict, optional): Declares the capabilities and parameters
              supported by each model. The configuration will be loaded and assigned to
              `model.profile` (e.g., `max_input_tokens`, `tool_calling`, etc.).
            - compatibility_options (CompatibilityOptions, optional): Compatibility
              options for the model provider. Only effective when `chat_model` is
              `"openai-compatible"`. Used to declare support for OpenAI-compatible features
              (e.g., `tool_choice` strategies, JSON mode, etc.).

    Raises:
        ValueError: If any of the providers are invalid

    Example:
        Register multiple providers at once::

            >>> from langchain_dev_utils.chat_models import batch_register_model_provider, load_chat_model
            >>> from langchain_core.language_models.fake_chat_models import FakeChatModel
            >>>
            >>> # Register multiple providers
            >>> batch_register_model_provider([
            ...     {
            ...         "provider_name": "fakechat",
            ...         "chat_model": FakeChatModel,
            ...     },
            ...     {
            ...         "provider_name": "vllm",
            ...         "chat_model": "openai-compatible",
            ...         "base_url": "http://localhost:8000/v1",
            ...     },
            ... ])
            >>>
            >>> # Use registered providers
            >>> model = load_chat_model("fakechat:fake-model")
            >>> model.invoke("Hello")
            >>>
            >>> model = load_chat_model("vllm:qwen3-4b")
            >>> model.invoke("Hello")
    """

    for provider in providers:
        register_model_provider(
            provider["provider_name"],
            provider["chat_model"],
            provider.get("base_url"),
            model_profiles=provider.get("model_profiles"),
            compatibility_options=provider.get("compatibility_options"),
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
