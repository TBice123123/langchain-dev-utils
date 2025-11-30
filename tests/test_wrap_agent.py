from typing import Any, cast

import pytest
from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import HumanMessage, ToolMessage

from langchain_dev_utils.agents.wrap import wrap_agent_as_tool


@tool
def get_time() -> str:
    """Get the current time."""
    return "The current time is 10:00 AM"


def process_input(request: str, runtime: ToolRuntime) -> str:
    return "<task_description>" + request + "</task_description>"


async def process_input_async(request: str, runtime: ToolRuntime) -> str:
    return "<task_description>" + request + "</task_description>"


def process_output(request: str, messages: list, runtime: ToolRuntime) -> str:
    assert request.startswith("<task_description>")
    assert request.endswith("</task_description>")
    return "<task_response>" + messages[-1].content + "</task_response>"


async def process_output_async(
    request: str, messages: list, runtime: ToolRuntime
) -> str:
    assert request.startswith("<task_description>")
    assert request.endswith("</task_description>")
    return "<task_response>" + messages[-1].content + "</task_response>"


def test_wrap_agent():
    agent = create_agent(model="deepseek:deepseek-chat", tools=[get_time])
    call_agent_tool = wrap_agent_as_tool(
        agent, "call_time_agent", "call the agent to query the time"
    )
    assert call_agent_tool.name == "call_time_agent"
    assert call_agent_tool.description == "call the agent to query the time"

    supervisor = create_agent(model="deepseek:deepseek-chat", tools=[call_agent_tool])
    response = supervisor.invoke(
        {"messages": [HumanMessage(content="What time is it now?")]}
    )

    msg = None
    for message in response["messages"]:
        if isinstance(message, ToolMessage) and message.name == "call_time_agent":
            msg = message
            break
    assert msg is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pre_input_hooks,post_output_hooks",
    [
        (
            process_input,
            process_output,
        ),
        (
            (process_input, process_input_async),
            (process_output, process_output_async),
        ),
    ],
)
async def test_wrap_agent_async(
    pre_input_hooks: Any,
    post_output_hooks: Any,
):
    agent = create_agent(
        model="deepseek:deepseek-chat", tools=[get_time], name="time_agent"
    )
    call_agent_tool = wrap_agent_as_tool(
        agent, pre_input_hooks=pre_input_hooks, post_output_hooks=post_output_hooks
    )
    assert call_agent_tool.name == "transfor_to_time_agent"
    assert call_agent_tool.description

    supervisor = create_agent(model="deepseek:deepseek-chat", tools=[call_agent_tool])
    response = await supervisor.ainvoke(
        {"messages": [HumanMessage(content="What time is it now?")]}
    )
    msg = None
    for message in response["messages"]:
        if (
            isinstance(message, ToolMessage)
            and message.name == "transfor_to_time_agent"
        ):
            msg = message
            break
    assert msg is not None

    assert cast(str, msg.content).startswith("<task_response>")
    assert cast(str, msg.content).endswith("</task_response>")
