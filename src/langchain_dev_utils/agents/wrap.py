import asyncio
from typing import Any, Awaitable, Callable, Optional

from langchain.tools import ToolRuntime
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.graph.state import CompiledStateGraph

from langchain_dev_utils.message_convert import format_sequence


def _process_input(request: str, runtime: ToolRuntime) -> str:
    return request


def _process_output(
    request: str, response: dict[str, Any], runtime: ToolRuntime
) -> Any:
    return response["messages"][-1].content


def wrap_agent_as_tool(
    agent: CompiledStateGraph,
    tool_name: Optional[str] = None,
    tool_description: Optional[str] = None,
    pre_input_hooks: Optional[
        tuple[
            Callable[[str, ToolRuntime], str | dict[str, Any]],
            Callable[[str, ToolRuntime], Awaitable[str | dict[str, Any]]],
        ]
        | Callable[[str, ToolRuntime], str | dict[str, Any]]
    ] = None,
    post_output_hooks: Optional[
        tuple[
            Callable[[str, dict[str, Any], ToolRuntime], Any],
            Callable[[str, dict[str, Any], ToolRuntime], Awaitable[Any]],
        ]
        | Callable[[str, dict[str, Any], ToolRuntime], Any]
    ] = None,
) -> BaseTool:
    """Wraps an agent as a tool

    Args:
        agent: The agent to wrap
        tool_name: The name of the tool
        tool_description: The description of the tool
        pre_input_hooks: Hooks to run before the input is processed
        post_output_hooks: Hooks to run after the output is processed

    Returns:
        BaseTool: The wrapped agent as a tool

    Example:
        >>> from langchain_dev_utils.agents import wrap_agent_as_tool, create_agent
        >>>
        >>> call_time_agent_tool = wrap_agent_as_tool(
        ...     time_agent,
        ...     tool_name="call_time_agent",
        ...     tool_description="Used to invoke the time sub-agent to perform time-related tasks"
        ... )
        >>>
        >>> agent = create_agent("vllm:qwen3-4b", tools=[call_time_agent_tool], name="agent")

        >>> response = agent.invoke({"messages": [HumanMessage(content="What time is it now?")]})
        >>> response
    """
    if agent.name is None:
        raise ValueError("Agent name must not be None")

    process_input = _process_input
    process_input_async = _process_input
    process_output = _process_output
    process_output_async = _process_output

    if pre_input_hooks:
        if isinstance(pre_input_hooks, tuple):
            process_input = pre_input_hooks[0]
            process_input_async = pre_input_hooks[1]
        else:
            process_input = pre_input_hooks
            process_input_async = pre_input_hooks

    if post_output_hooks:
        if isinstance(post_output_hooks, tuple):
            process_output = post_output_hooks[0]
            process_output_async = post_output_hooks[1]
        else:
            process_output = post_output_hooks
            process_output_async = post_output_hooks

    def call_agent(
        request: str,
        runtime: ToolRuntime,
    ):
        _processed_input = process_input(request, runtime) if process_input else request
        if isinstance(_processed_input, str):
            agent_input = {"messages": [HumanMessage(content=_processed_input)]}
        elif isinstance(_processed_input, dict):
            if "messages" not in _processed_input:
                raise ValueError("Agent input must contain 'messages' key")
            agent_input = _processed_input
        else:
            raise ValueError("Pre Hooks must return a string or a dict")

        response = agent.invoke(agent_input)

        response = (
            process_output(request, response, runtime)
            if process_output
            else response["messages"][-1].content
        )
        return response

    async def acall_agent(
        request: str,
        runtime: ToolRuntime,
    ):
        if asyncio.iscoroutinefunction(process_input_async):
            _processed_input = await process_input_async(request, runtime)
        else:
            _processed_input = (
                process_input_async(request, runtime)
                if process_input_async
                else request
            )

        if isinstance(_processed_input, str):
            agent_input = {"messages": [HumanMessage(content=_processed_input)]}
        elif isinstance(_processed_input, dict):
            if "messages" not in _processed_input:
                raise ValueError("Agent input must contain 'messages' key")
            agent_input = _processed_input
        else:
            raise ValueError("Pre Hooks must return a string or a dict")

        response = await agent.ainvoke(agent_input)

        if asyncio.iscoroutinefunction(process_output_async):
            response = await process_output_async(request, response, runtime)
        else:
            response = (
                process_output(request, response, runtime)
                if process_output
                else response["messages"][-1].content
            )

        return response

    if tool_name is None:
        tool_name = f"transfor_to_{agent.name}"
        if not tool_name.endswith("_agent"):
            tool_name += "_agent"

    if tool_description is None:
        tool_description = f"This tool transforms input to {agent.name}"

    return StructuredTool.from_function(
        func=call_agent,
        coroutine=acall_agent,
        name=tool_name,
        description=tool_description,
    )


def wrap_all_agents_as_tool(
    agents: list[CompiledStateGraph],
    tool_name: Optional[str] = None,
    tool_description: Optional[str] = None,
    pre_input_hooks: Optional[
        tuple[
            Callable[[str, ToolRuntime], str | dict[str, Any]],
            Callable[[str, ToolRuntime], Awaitable[str | dict[str, Any]]],
        ]
        | Callable[[str, ToolRuntime], str | dict[str, Any]]
    ] = None,
    post_output_hooks: Optional[
        tuple[
            Callable[[str, dict[str, Any], ToolRuntime], Any],
            Callable[[str, dict[str, Any], ToolRuntime], Awaitable[Any]],
        ]
        | Callable[[str, dict[str, Any], ToolRuntime], Any]
    ] = None,
) -> BaseTool:
    """Wraps all agents as single tool

    Args:
        agents: The agents to wrap
        tool_name: The name of the tool, default to "task"
        tool_description: The description of the tool
        pre_input_hooks: Hooks to run before the input is processed
        post_output_hooks: Hooks to run after the output is processed

    Returns:
        BaseTool: The wrapped agents as single tool

    Example:
        >>> from langchain_dev_utils.agents import wrap_all_agents_as_tool, create_agent
        >>>
        >>> call_agent_tool = wrap_all_agents_as_tool(
        ...     [time_agent,weather_agent],
        ...     tool_name="call_sub_agents",
        ...     tool_description="Used to invoke the sub-agents to perform tasks"
        ... )
        >>>
        >>> agent = create_agent("vllm:qwen3-4b", tools=[call_sub_agents_tool], name="agent")

        >>> response = agent.invoke({"messages": [HumanMessage(content="What time is it now?")]})
        >>> response
    """
    if len(agents) <= 1:
        raise ValueError("At least more than one agent must be provided")

    agents_map = {}

    for agent in agents:
        if agent.name is None:
            raise ValueError("Agent name must not be provided")
        if agent.name in agents_map:
            raise ValueError("Agent name must be unique")
        agents_map[agent.name] = agent

    process_input = _process_input
    process_input_async = _process_input
    process_output = _process_output
    process_output_async = _process_output

    if pre_input_hooks:
        if isinstance(pre_input_hooks, tuple):
            process_input = pre_input_hooks[0]
            process_input_async = pre_input_hooks[1]
        else:
            process_input = pre_input_hooks
            process_input_async = pre_input_hooks

    if post_output_hooks:
        if isinstance(post_output_hooks, tuple):
            process_output = post_output_hooks[0]
            process_output_async = post_output_hooks[1]
        else:
            process_output = post_output_hooks
            process_output_async = post_output_hooks

    def call_agent(
        agent_name: str,
        description: str,
        runtime: ToolRuntime,
    ):
        if agent_name not in agents_map:
            raise ValueError(f"Agent {agent_name} not found")

        _processed_input = (
            process_input(description, runtime) if process_input else description
        )
        if isinstance(_processed_input, str):
            agent_input = {"messages": [HumanMessage(content=_processed_input)]}
        elif isinstance(_processed_input, dict):
            if "messages" not in _processed_input:
                raise ValueError("Agent input must contain 'messages' key")
            agent_input = _processed_input
        else:
            raise ValueError("Pre Hooks must return str or dict")

        response = agent.invoke(agent_input)

        response = (
            process_output(description, response, runtime)
            if process_output
            else response["messages"][-1].content
        )
        return response

    async def acall_agent(
        agent_name: str,
        description: str,
        runtime: ToolRuntime,
    ):
        if agent_name not in agents_map:
            raise ValueError(f"Agent {agent_name} not found")

        if asyncio.iscoroutinefunction(process_input_async):
            _processed_input = await process_input_async(description, runtime)
        else:
            _processed_input = (
                process_input_async(description, runtime)
                if process_input_async
                else description
            )

        if isinstance(_processed_input, str):
            agent_input = {"messages": [HumanMessage(content=_processed_input)]}
        elif isinstance(_processed_input, dict):
            if "messages" not in _processed_input:
                raise ValueError("Agent input must contain 'messages' key")
            agent_input = _processed_input
        else:
            raise ValueError("Pre Hooks must return str or dict")

        response = await agents_map[agent_name].ainvoke(agent_input)

        if asyncio.iscoroutinefunction(process_output_async):
            response = await process_output_async(description, response, runtime)
        else:
            response = (
                process_output(description, response, runtime)
                if process_output
                else response["messages"][-1].content
            )

        return response

    if tool_name is None:
        tool_name = "task"

    if tool_description is None:
        tool_description = (
            "Launch an ephemeral subagent for a task.\nAvailable agents:\n "
            + format_sequence(list(agents_map.keys()), with_num=True)
        )
    return StructuredTool.from_function(
        func=call_agent,
        coroutine=acall_agent,
        name=tool_name,
        description=tool_description,
    )
