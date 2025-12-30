from typing import Any, Awaitable, Callable, Literal

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
    tools: NotRequired[list[BaseTool | dict[str, Any]]]
    default: NotRequired[bool]
    handoffs: list[str] | Literal["all"]


def _create_handoffs_tool(agent_name: str, tool_description: Optional[str] = None):
    """Create a tool for handoffs to a specified agent.

    Args:
        agent_name (str): The name of the agent to transfer to.

    Returns:
        BaseTool: A tool instance for handoffs to the specified agent.
    """

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


def _transform_agent_config(
    config: dict[str, AgentConfig],
    handoffs_tools: list[BaseTool],
) -> dict[str, AgentConfig]:
    """Transform the agent config to add handoffs tools.

    Args:
        config (dict[str, AgentConfig]): The agent config.
        handoffs_tools (list[BaseTool]): The list of handoffs tools.

    Returns:
        dict[str, AgentConfig]: The transformed agent config.
    """

    new_config = {}
    for agent_name, _cfg in config.items():
        new_config[agent_name] = {}

        if "model" in _cfg:
            new_config[agent_name]["model"] = _cfg["model"]
        if "prompt" in _cfg:
            new_config[agent_name]["prompt"] = _cfg["prompt"]
        if "default" in _cfg:
            new_config[agent_name]["default"] = _cfg["default"]
        if "tools" in _cfg:
            new_config[agent_name]["tools"] = _cfg["tools"]

        handoffs = _cfg.get("handoffs", [])
        if handoffs == "all":
            handoff_tools = [
                handoff_tool
                for handoff_tool in handoffs_tools
                if handoff_tool.name != f"transfer_to_{agent_name}"
            ]
        else:
            if not isinstance(handoffs, list):
                raise ValueError(
                    f"handoffs for agent {agent_name} must be a list of agent names or 'all'"
                )

            handoff_tools = [
                handoff_tool
                for handoff_tool in handoffs_tools
                if handoff_tool.name
                in [
                    f"transfer_to_{_handoff_agent_name}"
                    for _handoff_agent_name in handoffs
                ]
            ]

        new_config[agent_name]["tools"] = [
            *new_config[agent_name].get("tools", []),
            *handoff_tools,
        ]
    return new_config


class HandoffAgentMiddleware(AgentMiddleware):
    """Agent middleware for switching between multiple agents.
    This middleware dynamically replaces model call parameters based on the currently active agent configuration, enabling seamless switching between different agents.

    Args:
        agents_config (dict[str, AgentConfig]): A dictionary of agent configurations.
        custom_handoffs_tool_descriptions (Optional[dict[str, str]]): A dictionary of custom tool descriptions for handoffs tools. Defaults to None.

    Examples:
        ```python
        from langchain_dev_utils.agents.middleware import HandoffAgentMiddleware
        middleware = HandoffAgentMiddleware(agents_config)
        ```
    """

    state_schema = MultiAgentState

    def __init__(
        self,
        agents_config: dict[str, AgentConfig],
        custom_handoffs_tool_descriptions: Optional[dict[str, str]] = None,
    ) -> None:
        default_agent_name = _get_default_active_agent(agents_config)
        if default_agent_name is None:
            raise ValueError(
                "No default agent found, you must set one by set default=True"
            )

        if custom_handoffs_tool_descriptions is None:
            custom_handoffs_tool_descriptions = {}

        handoffs_tools = [
            _create_handoffs_tool(
                agent_name,
                custom_handoffs_tool_descriptions.get(agent_name),
            )
            for agent_name in agents_config.keys()
        ]
        self.default_agent_name = default_agent_name
        self.agents_config = _transform_agent_config(
            agents_config,
            handoffs_tools,
        )
        self.tools = handoffs_tools

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
