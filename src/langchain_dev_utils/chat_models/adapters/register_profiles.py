from typing import Any

_PROFILES = {}


def _register_profile_with_provider(
    provider_name: str, profile: dict[str, Any]
) -> None:
    _PROFILES.update({provider_name: profile})


def _get_profile_by_provider_and_model(
    provider_name: str, model_name: str
) -> dict[str, Any]:
    return _PROFILES.get(provider_name, {}).get(model_name, {})
