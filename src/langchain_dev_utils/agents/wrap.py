import asyncio
from typing import Any, Awaitable, Optional, Callable, cast

from langchain.tools import ToolRuntime
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field


class AgentToolInput(BaseModel):
    request: str = Field(description="The input to the agent")


def _process_input(request: str, runtime: ToolRuntime) -> str:
    return request


def _process_output(
    request: str, response: list[AnyMessage], runtime: ToolRuntime
) -> Any:
    return response[-1].content


def wrap_agent_as_tool(
    agent: CompiledStateGraph,
    tool_name: Optional[str] = None,
    tool_description: Optional[str] = None,
    pre_input_hooks: Optional[
        tuple[
            Callable[[str, ToolRuntime], str],
            Callable[[str, ToolRuntime], Awaitable[str]],
        ]
        | Callable[[str, ToolRuntime], str]
    ] = None,
    post_output_hooks: Optional[
        tuple[
            Callable[[str, list[AnyMessage], ToolRuntime], Any],
            Callable[[str, list[AnyMessage], ToolRuntime], Awaitable[Any]],
        ]
        | Callable[[str, list[AnyMessage], ToolRuntime], Any]
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
        >>> import datetime
        >>> from langchain_core.messages import HumanMessage
        >>> from langchain_core.tools import tool
        >>> from langchain_dev_utils.agents import wrap_agent_as_tool, create_agent

        >>> @tool
        ... def get_current_time() -> str:
        ...     "\"\"Get the current timestamp.\"\"\"
        ...     return str(datetime.datetime.now().timestamp())

        >>> # Define an agent for querying the time
        >>> time_agent = create_agent(
        ...     "vllm:qwen3-4b", tools=[get_current_time], name="time_agent"
        ... )
        >>> tool = wrap_agent_as_tool(
        ...     time_agent, tool_name="call_time_agent", tool_description="Invoke the time agent"
        ... )
        >>> print(tool)

        >>> # Use it as a tool
        >>> agent = create_agent("vllm:qwen3-4b", tools=[tool], name="agent")

        >>> response = agent.invoke({"messages": [HumanMessage(content="What time is it now?")]})
        >>> print(response)
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
    ) -> str:
        request = process_input(request, runtime) if process_input else request

        messages = [HumanMessage(content=request)]
        response = agent.invoke({"messages": messages})

        response = process_output(request, response["messages"], runtime)
        return response

    async def acall_agent(
        request: str,
        runtime: ToolRuntime,
    ) -> str:
        if asyncio.iscoroutinefunction(process_input_async):
            request = await process_input_async(request, runtime)
        else:
            request = cast(str, process_input_async(request, runtime))

        messages = [HumanMessage(content=request)]
        response = await agent.ainvoke({"messages": messages})

        if asyncio.iscoroutinefunction(process_output_async):
            response = await process_output_async(
                request, response["messages"], runtime
            )
        else:
            response = process_output(request, response["messages"], runtime)

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
