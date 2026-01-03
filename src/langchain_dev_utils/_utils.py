from importlib import util
from typing import Literal, Optional

from pydantic import BaseModel


def _check_pkg_install(
    pkg: Literal["langchain_openai", "json_repair"],
) -> None:
    if not util.find_spec(pkg):
        if pkg == "langchain_openai":
            msg = "Please install langchain_dev_utils[standard],when use 'openai-compatible'"
        else:
            msg = "Please install langchain_dev_utils[standard] to use ToolCallRepairMiddleware."
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


def _validate_base_url(base_url: Optional[str] = None) -> None:
    """Validate base URL format.

    Args:
        base_url: Base URL to validate

    Raises:
        ValueError: If base URL is not a valid HTTP or HTTPS URL
    """
    if base_url is None:
        return

    from urllib.parse import urlparse

    parsed = urlparse(base_url.strip())

    if not parsed.scheme or not parsed.netloc:
        raise ValueError(
            f"base_url must be a valid HTTP or HTTPS URL. Received: {base_url}"
        )

    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"base_url must use HTTP or HTTPS protocol. Received: {parsed.scheme}"
        )


def _validate_model_cls_name(model_cls_name: str) -> None:
    """Validate model class name follows Python naming conventions.

    Args:
        model_cls_name: Class name to validate

    Raises:
        ValueError: If class name is invalid
    """
    if not model_cls_name:
        raise ValueError("model_cls_name cannot be empty")

    if not model_cls_name[0].isalpha():
        raise ValueError(
            f"model_cls_name must start with a letter. Received: {model_cls_name}"
        )

    if not all(c.isalnum() or c == "_" for c in model_cls_name):
        raise ValueError(
            f"model_cls_name can only contain letters, numbers, and underscores. Received: {model_cls_name}"
        )

    if model_cls_name[0].islower():
        raise ValueError(
            f"model_cls_name should start with an uppercase letter (PEP 8). Received: {model_cls_name}"
        )
