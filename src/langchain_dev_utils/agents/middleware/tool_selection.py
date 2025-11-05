from typing import Optional

from langchain.agents.middleware.tool_selection import (
    LLMToolSelectorMiddleware as _LLMToolSelectorMiddleware,
)

from langchain_dev_utils.chat_models.base import load_chat_model


class LLMToolSelectorMiddleware(_LLMToolSelectorMiddleware):
    """Intelligent tool selection middleware using LLM to filter relevant tools.

    This middleware leverages a language model to analyze user queries and select
    only the most relevant tools from a potentially large toolset. This optimization
    reduces token consumption and improves model performance by focusing on appropriate tools.

    The selection process analyzes the user's request and matches it against available
    tool descriptions to determine relevance.

    Args:
        model: String identifier for the model to use for tool selection.
            Must be a valid model identifier that can be loaded by load_chat_model().
        system_prompt: Custom instructions for the selection model. If not provided,
            uses the default selection prompt from the parent class.
        max_tools: Maximum number of tools to select and pass to the main model.
            If the LLM selects more tools than this limit, only the first max_tools
            tools will be used. If None, no limit is applied.
        always_include: List of tool names that must always be included in the
            selection regardless of the LLM's decision. These tools do not count
            against the max_tools limit.

    Examples:
        Basic usage with tool limit:
        ```python
        from langchain_dev_utils.agents.middleware import LLMToolSelectorMiddleware

        middleware = LLMToolSelectorMiddleware(
            model="vllm:qwen3-4b",
            max_tools=3
        )
        ```

        With always-included tools:
        ```python
        middleware = LLMToolSelectorMiddleware(
            model="vllm:qwen3-4b",
            max_tools=5,
            always_include=["search", "calculator"]
        )
        ```

        With custom system prompt:
        ```python
        custom_prompt = "Select tools that can help answer user questions about data."
        middleware = LLMToolSelectorMiddleware(
            model="vllm:qwen3-4b",
            system_prompt=custom_prompt
        )
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
