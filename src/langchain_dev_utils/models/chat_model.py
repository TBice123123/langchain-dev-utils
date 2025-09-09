import os
from typing import Any, Optional, Union, cast

from langchain.chat_models.base import (
    BaseChatModel,
    _SUPPORTED_PROVIDERS,
    _init_chat_model_helper,
    init_chat_model,
)

_MODEL_PROVIDERS_DICT = {}


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
        if isinstance(chat_model, str):
            if not (api_key := kwargs.get("api_key")):
                api_key = os.getenv(f"{model_provider.upper()}_API_KEY")
                if not api_key:
                    raise ValueError(
                        f"API key for {model_provider} not found. Please set it in the environment."
                    )
                kwargs["api_key"] = api_key
            base_url = _MODEL_PROVIDERS_DICT[model_provider]["base_url"]
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

    Args:
        provider_name: Name of the provider to register
        chat_model: Either a BaseChatModel class or a string identifier for a supported provider
        base_url: Optional base URL for API endpoints (required when chat_model is a string)

    Raises:
        ValueError: If base_url is not provided when chat_model is a string,
                   or if chat_model string is not in supported providers
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


def load_chat_model(
    model: str,
    *,
    model_provider: Optional[str] = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Load a chat model.

    Args:
        model: Model name
        model_provider: Optional provider name
        **kwargs: Additional arguments for model initialization

    Returns:
        BaseChatModel: Initialized chat model instance
    """
    return _load_chat_model_helper(
        cast(str, model),
        model_provider=model_provider,
        **kwargs,
    )
