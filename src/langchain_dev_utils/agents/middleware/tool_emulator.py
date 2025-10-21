from langchain.agents.middleware.tool_emulator import (
    LLMToolEmulator as _LLMToolEmulator,
)
from langchain_core.tools import BaseTool

from langchain_dev_utils.chat_models.base import load_chat_model


class LLMToolEmulator(_LLMToolEmulator):
    """Middleware that emulates specified tools using an LLM instead of executing them.

    This middleware allows selective emulation of tools for testing purposes.
    By default (when tools=None), all tools are emulated. You can specify which
    tools to emulate by passing a list of tool names or BaseTool instances.

    Examples:
        Emulate all tools (default behavior):
        ```python
        from langchain_dev_utils.agents import create_agent
        from langchain_dev_utils.agents.middleware import LLMToolEmulator

        middleware = LLMToolEmulator(
            model="vllm:qwen3-4b"
        )

        agent = create_agent(
            model="vllm:qwen3-4b",
            tools=[get_weather, get_user_location, calculator],
            middleware=[middleware],
        )
        ```

        Emulate specific tools by name:
        ```python
        middleware = LLMToolEmulator(model="vllm:qwen3-4b", tools=["get_weather", "get_user_location"])
        ```

        Emulate specific tools by passing tool instances:
        ```python
        middleware = LLMToolEmulator(model="vllm:qwen3-4b", tools=[get_weather, get_user_location])
        ```
    """

    def __init__(
        self,
        *,
        model: str,
        tools: list[str | BaseTool] | None = None,
    ) -> None:
        """Initialize the tool emulator.

        Args:
            tools: List of tool names (str) or BaseTool instances to emulate.
                If None (default), ALL tools will be emulated.
                If empty list, no tools will be emulated.
            model: Model to use for emulation. Must be a string identifier.
        """
        chat_model = load_chat_model(model)
        super().__init__(
            model=chat_model,
            tools=tools,
        )
