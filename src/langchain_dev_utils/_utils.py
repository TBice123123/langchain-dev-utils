from importlib import util
from typing import Literal, Optional, cast

from langgraph.graph import StateGraph
from langgraph.graph.state import StateNode
from pydantic import BaseModel


def _transform_node_to_tuple(
    node: StateNode | tuple[str, StateNode],
) -> tuple[str, StateNode]:
    if not isinstance(node, tuple):
        if isinstance(node, StateGraph):
            node = node.compile()
            name = node.name
            return name, node
        name = cast(str, getattr(node, "name", getattr(node, "__name__", None)))
        if name is None:
            raise ValueError("Node name must be provided if action is not a function")
        return name, node
    else:
        return node


def _check_pkg_install(
    pkg: Literal["langchain_openai", "json_repair", "jinja2"],
) -> None:
    if not util.find_spec(pkg):
        if pkg == "langchain_openai":
            msg = "Please install langchain_dev_utils[standard],when use 'openai-compatible'"
        elif pkg == "json_repair":
            msg = "Please install langchain_dev_utils[standard] to use ToolCallRepairMiddleware."
        else:
            msg = "Please install langchain_dev_utils[standard] to use FormatPromptMiddleware."
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

    if len(model_cls_name) > 30:
        raise ValueError(
            f"model_cls_name must be 30 characters or fewer. Received: {model_cls_name}"
        )


def _validate_provider_name(provider_name: str) -> None:
    """Validate provider name follows Python naming conventions.

    Args:
        provider_name: Provider name to validate

    Raises:
        ValueError: If provider name is invalid
    """
    if not provider_name:
        raise ValueError("provider_name cannot be empty")

    if not provider_name[0].isalnum():
        raise ValueError(
            f"provider_name must start with a letter or number. Received: {provider_name}"
        )

    if not all(c.isalnum() or c == "_" for c in provider_name):
        raise ValueError(
            f"provider_name can only contain letters, numbers, underscores. Received: {provider_name}"
        )

    if len(provider_name) > 20:
        raise ValueError(
            f"provider_name must be 20 characters or fewer. Received: {provider_name}"
        )
