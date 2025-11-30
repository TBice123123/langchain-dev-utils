from importlib import util

from pydantic import BaseModel


def _check_langchain_openai_install() -> None:
    if not util.find_spec("langchain_openai"):
        msg = (
            "Please install langchain_dev_utils[standard],when use 'openai-compatible'"
        )
        raise ImportError(msg)


def _get_base_url_field_name(model_cls: type[BaseModel]) -> str | None:
    """
    Return 'base_url' if the model has a field named or aliased as 'base_url',
    else return 'api_base' if it has a field named or aliased as 'api_base',
    else return None.
    The return value is always either 'base_url', 'api_base', or None.
    """
    model_fields = model_cls.model_fields

    # try model_fields first
    if "base_url" in model_fields:
        return "base_url"

    if "api_base" in model_fields:
        return "api_base"

    # then try aliases
    for field_info in model_fields.values():
        if field_info.alias == "base_url":
            return "base_url"

    for field_info in model_fields.values():
        if field_info.alias == "api_base":
            return "api_base"

    return None
