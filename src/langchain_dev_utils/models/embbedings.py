import os
from typing import Any, Optional, Union

from langchain.embeddings.base import Embeddings, _SUPPORTED_PROVIDERS, init_embeddings
from langchain_core.runnables import Runnable


_EMBEDDINGS_PROVIDERS_DICT = {}


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
    embeddings_model: Union[type[Embeddings], str],
    base_url: Optional[str] = None,
):
    """Register an embeddings provider.

    Args:
        provider_name: Name of the provider to register
        embeddings_model: Either an Embeddings class or a string identifier for a supported provider
        base_url: Optional base URL for API endpoints (required when embeddings_model is a string)

    Raises:
        ValueError: If base_url is not provided when embeddings_model is a string
    """
    if isinstance(embeddings_model, str):
        if base_url is None:
            raise ValueError(
                "base_url must be provided when embeddings_model is a string"
            )

        if embeddings_model not in _SUPPORTED_PROVIDERS:
            raise ValueError(
                f"when embeddings_model is a string, the value must be one of {_SUPPORTED_PROVIDERS}"
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


def load_embeddings(
    model: str,
    *,
    provider: Optional[str] = None,
    **kwargs: Any,
) -> Union[Embeddings, Runnable[Any, list[float]]]:
    """Load embeddings model.

    Args:
        model: Model name in format 'provider:model-name' if provider not specified separately
        provider: Optional provider name (if not included in model parameter)
        **kwargs: Additional arguments for model initialization

    Returns:
        Union[Embeddings, Runnable[Any, list[float]]]: Initialized embeddings model instance

    Raises:
        ValueError: If provider is not registered or API key is not found
    """
    if provider is None:
        provider, model = _parse_model_string(model)
    if provider not in _EMBEDDINGS_PROVIDERS_DICT:
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
            if embeddings == "openai":
                kwargs["check_embedding_ctx_length"] = False

        return init_embeddings(
            model=model,
            provider=embeddings,
            base_url=_EMBEDDINGS_PROVIDERS_DICT[provider]["base_url"],
            **kwargs,
        )
    else:
        return embeddings(model=model, **kwargs)
