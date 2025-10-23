from typing import Optional

from langchain.agents.middleware.tool_selection import (
    LLMToolSelectorMiddleware as _LLMToolSelectorMiddleware,
)

from langchain_dev_utils.chat_models.base import load_chat_model


class LLMToolSelectorMiddleware(_LLMToolSelectorMiddleware):
    """Uses an LLM to select relevant tools before calling the main model.

    When an agent has many tools available, this middleware filters them down
    to only the most relevant ones for the user's query. This reduces token usage
    and helps the main model focus on the right tools.

    Args:
        model: Model to use for selection. Only string identifiers are supported.
        system_prompt: Instructions for the selection model.
        max_tools: Maximum number of tools to select. If the model selects more,
            only the first max_tools will be used. No limit if not specified.
        always_include: Tool names to always include regardless of selection.
            These do not count against the max_tools limit.

    Examples:
        Limit to 3 tools:
        ```python
        from langchain_dev_utils.agents.middleware import LLMToolSelectorMiddleware

        middleware = LLMToolSelectorMiddleware(model="vllm:qwen3-4b", max_tools=3)
        ```
    """

    def __init__(
        self,
        *,
        model: str,
        system_prompt: Optional[str] = None,
        max_tools: Optional[int] = None,
        always_include: Optional[list[str]] = None,
    ) -> None:
        chat_model = load_chat_model(model)

        tool_selector_kwargs = {}
        if system_prompt is not None:
            tool_selector_kwargs["system_prompt"] = system_prompt
        if max_tools is not None:
            tool_selector_kwargs["max_tools"] = max_tools
        if always_include is not None:
            tool_selector_kwargs["always_include"] = always_include
        super().__init__(
            model=chat_model,
            **tool_selector_kwargs,
        )
