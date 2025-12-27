from typing import Any, Awaitable, Callable

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ModelCallResult
from langchain.tools import BaseTool, ToolRuntime, tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.types import Command
from typing_extensions import NotRequired, Optional, TypedDict

from langchain_dev_utils.chat_models import load_chat_model


class MultiAgentState(AgentState):
    active_agent: NotRequired[str]


class AgentConfig(TypedDict):
    model: NotRequired[str | BaseChatModel]
    prompt: str | SystemMessage
    tools: list[BaseTool | dict[str, Any]]
    default: NotRequired[bool]


def create_handoffs_tool(
    agent_name: str,
    tool_name: Optional[str] = None,
    tool_description: Optional[str] = None,
):
    """Create a tool for handoffs to a specified agent.

    Args:
        agent_name (str): The name of the agent to transfer to.
        tool_name (Optional[str], optional): The name of the tool. Defaults to None.
        tool_description (Optional[str], optional): The description of the tool. Defaults to None.

    Returns:
        BaseTool: A tool instance for handoffs to the specified agent.

    Example:
        Basic usage
        >>> from langchain_dev_utils.agents.middleware import create_handoffs_tool
        >>> handoffs_tool = create_handoffs_tool("time_agent")
    """
    if tool_name is None:
        tool_name = f"transfer_to_{agent_name}"
        if not tool_name.endswith("_agent"):
            tool_name += "_agent"

    if tool_description is None:
        tool_description = f"Transfer to the {agent_name}"

    @tool(name_or_callable=tool_name, description=tool_description)
    def handoffs_tool(runtime: ToolRuntime) -> Command:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Transferred to {agent_name}",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
                "active_agent": agent_name,
            }
        )

    return handoffs_tool


def _get_default_active_agent(state: dict[str, AgentConfig]) -> Optional[str]:
    for agent_name, config in state.items():
        if config.get("default", False):
            return agent_name
    return None


class HandoffsAgentMiddleware(AgentMiddleware):
    """Agent middleware for switching between multiple agents.
    This middleware dynamically replaces model call parameters based on the currently active agent configuration, enabling seamless switching between different agents.

    Args:
        agents_config (dict[str, AgentConfig]): A dictionary of agent configurations.

    Examples:
        ```python
        from langchain_dev_utils.agents.middleware import HandoffsAgentMiddleware
        middleware = HandoffsAgentMiddleware(agents_config)
        ```
    """

    state_schema = MultiAgentState

    def __init__(self, agents_config: dict[str, AgentConfig]):
        default_agent_name = _get_default_active_agent(agents_config)
        if default_agent_name is None:
            raise ValueError(
                "No default agent found, you must set one by set default=True"
            )
        self.default_agent_name = default_agent_name
        self.agents_config = agents_config

    def _get_active_agent_config(self, request: ModelRequest) -> dict[str, Any]:
        active_agent_name = request.state.get("active_agent", self.default_agent_name)

        _config = self.agents_config[active_agent_name]

        params = {}
        if _config.get("model"):
            model = _config.get("model")
            if isinstance(model, str):
                model = load_chat_model(model)
            params["model"] = model
        if _config.get("prompt"):
            params["system_prompt"] = _config.get("prompt")
        if _config.get("tools"):
            params["tools"] = _config.get("tools")
        return params

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelCallResult:
        override_kwargs = self._get_active_agent_config(request)
        if override_kwargs:
            return handler(request.override(**override_kwargs))
        else:
            return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        override_kwargs = self._get_active_agent_config(request)
        if override_kwargs:
            return await handler(request.override(**override_kwargs))
        else:
            return await handler(request)
