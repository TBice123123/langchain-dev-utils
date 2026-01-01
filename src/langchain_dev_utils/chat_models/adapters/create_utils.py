from typing import Any, Optional, cast

from langchain_core.utils import from_env

from langchain_dev_utils._utils import _check_pkg_install

from ..types import CompatibilityOptions


def create_openai_compatible_model(
    model_provider: str,
    base_url: Optional[str] = None,
    compatibility_options: Optional[CompatibilityOptions] = None,
    model_profiles: Optional[dict[str, dict[str, Any]]] = None,
    chat_model_cls_name: Optional[str] = None,
):
    """Factory function for creating provider-specific OpenAI-compatible model classes.

    Dynamically generates model classes for different OpenAI-compatible providers,
    configuring environment variable mappings and default base URLs specific to each provider.

    Args:
        model_provider (str): Identifier for the OpenAI-compatible provider (e.g. `vllm`, `moonshot`)
        base_url (Optional[str], optional): Default API base URL for the provider. Defaults to None. If not provided, will try to use the environment variable.
        compatibility_options (Optional[CompatibilityOptions], optional): Optional configuration for compatibility options with the provider. Defaults to None.
        model_profiles (Optional[dict[str, dict[str, Any]]], optional): Optional model profiles for the provider. Defaults to None.
        chat_model_cls_name (Optional[str], optional): Optional custom class name for the generated model. Defaults to None.
    Returns:
        Type[_BaseChatOpenAICompatible]: Configured model class ready for instantiation with provider-specific settings

    Examples:
        >>> from langchain_dev_utils.chat_models.adapters import create_openai_compatible_chat_model
        >>> ChatVLLM = create_openai_compatible_chat_model(
        ...     "vllm",
        ...     base_url="http://localhost:8000",
        ...     chat_model_cls_name="ChatVLLM",
        ... )
        >>> model = ChatVLLM(model="qwen3-4b")
        >>> model.invoke("hello")
    """
    _check_pkg_install("langchain_openai")
    from .openai_compatible import _create_openai_compatible_model

    base_url = (
        base_url or from_env(f"{model_provider.upper()}_API_BASE", default=None)()
    )
    return _create_openai_compatible_model(
        chat_model_cls_name=chat_model_cls_name,
        provider=model_provider,
        base_url=cast(str, base_url),
        compatibility_options=compatibility_options,
        profiles=model_profiles,
    )
