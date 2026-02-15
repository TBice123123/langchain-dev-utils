from typing import Awaitable, Callable, Literal, cast

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ModelCallResult
from langchain.tools import BaseTool, ToolRuntime, tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.types import Command
from typing_extensions import NotRequired, Optional, TypedDict

from langchain_dev_utils._utils import _duplicate_tools, _merge_tools
from langchain_dev_utils.chat_models import load_chat_model


class MultiAgentState(AgentState):
    active_agent: NotRequired[str]


class AgentConfig(TypedDict):
    model: NotRequired[str | BaseChatModel]
    prompt: str | SystemMessage
    tools: NotRequired[list[BaseTool]]
    default: NotRequired[bool]
    handoffs: list[str] | Literal["all"]


def _create_handoffs_tool(
    agent_name: str, tool_description: Optional[str] = None
) -> BaseTool:
    """Create a tool for handoffs to a specified agent.

    Args:
        agent_name (str): The name of the agent to transfer to.

    Returns:
        BaseTool: The tool instance for handoffs to the specified agent.
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
    handoffs_tool_map: dict[str, BaseTool],
) -> dict[str, AgentConfig]:
    """Transform the agent config to add handoffs tools.

    Args:
        config (dict[str, AgentConfig]): The agent config.
        handoffs_tool_map (dict[str, BaseTool]): The dict of {agent name: handoffs tool}.

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
                for transfer_to_agent, handoff_tool in handoffs_tool_map.items()
                if transfer_to_agent != agent_name
            ]
        else:
            if not isinstance(handoffs, list):
                raise ValueError(
                    f"handoffs for agent {agent_name} must be a list of agent names or 'all'"
                )

            if agent_name in handoffs:
                raise ValueError(f"agent {agent_name} cannot handoff to itself")

            handoff_tools = [
                handoff_tool
                for transfer_to_agent, handoff_tool in handoffs_tool_map.items()
                if transfer_to_agent in handoffs
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
        handoffs_tool_overrides (Optional[dict[str, BaseTool]]): A dictionary of handoffs tools to override. Defaults to None.

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
        handoffs_tool_overrides: Optional[dict[str, BaseTool]] = None,
    ) -> None:
        default_agent_name = _get_default_active_agent(agents_config)
        if default_agent_name is None:
            raise ValueError(
                "No default agent found, you must set one by set default=True"
            )

        if custom_handoffs_tool_descriptions is None:
            custom_handoffs_tool_descriptions = {}

        if handoffs_tool_overrides is None:
            handoffs_tool_overrides = {}

        handoffs_tool_map = {}

        for agent_name in agents_config.keys():
            if not handoffs_tool_overrides.get(agent_name):
                handoffs_tool_map[agent_name] = _create_handoffs_tool(
                    agent_name,
                    custom_handoffs_tool_descriptions.get(agent_name),
                )

            else:
                handoffs_tool_map[agent_name] = cast(
                    BaseTool, handoffs_tool_overrides.get(agent_name)
                )

        self.default_agent_name = default_agent_name
        self.agents_config = _transform_agent_config(
            agents_config,
            handoffs_tool_map,
        )

        all_tools = [
            *handoffs_tool_map.values(),
        ]

        for agent_name in agents_config:
            tools = agents_config.get(agent_name, {}).get("tools", [])
            all_tools.extend(tools)

        self.tools = _merge_tools(all_tools)

    def _get_override_request(self, request: ModelRequest) -> ModelRequest:
        active_agent_name = request.state.get("active_agent", self.default_agent_name)

        _config = self.agents_config[active_agent_name]

        params = {}
        if model := _config.get("model"):
            if isinstance(model, str):
                model = load_chat_model(model)
            params["model"] = model
        if prompt := _config.get("prompt"):
            params["system_prompt"] = prompt
        if tools := _config.get("tools"):
            params["tools"] = _duplicate_tools(tools)

        if params:
            return request.override(**params)
        else:
            return request

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelCallResult:
        override_request = self._get_override_request(request)
        return handler(override_request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        override_request = self._get_override_request(request)
        return await handler(override_request)
