from langchain.agents import create_agent
import pytest

from langchain_dev_utils.agents.wrap import wrap_agent_as_tool


def test_wrap_agent():
    agent = create_agent(model="deepseek:deepseek-chat")
    tool = wrap_agent_as_tool(agent, "call_agent", "This tool calls the agent")
    assert tool.name == "call_agent"
    assert tool.description == "This tool calls the agent"
    assert tool.invoke({"request": "Hello, how are you?"})


@pytest.mark.asyncio
async def test_wrap_agent_async():
    agent = create_agent(model="deepseek:deepseek-chat")
    tool = wrap_agent_as_tool(agent)
    tool_name = f"transform_to_{agent.name}"
    if not tool_name.endswith("_agent"):
        tool_name += "_agent"
    assert tool.name == tool_name
    assert tool.description
    assert await tool.ainvoke({"request": "Hello, how are you?"})
