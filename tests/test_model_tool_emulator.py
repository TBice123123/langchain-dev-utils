from langchain.tools import tool
from langchain_qwq import ChatQwen
from langchain_core.messages import HumanMessage, ToolMessage
import pytest

from langchain_dev_utils.agents import create_agent
from langchain_dev_utils.agents.middleware import LLMToolEmulator
from langchain_dev_utils.chat_models import register_model_provider
from dotenv import load_dotenv

load_dotenv()
register_model_provider(
    "dashscope",
    ChatQwen,
)


def test_model_tool_emulator():
    middleware = LLMToolEmulator(model="dashscope:qwen-flash")

    @tool
    def get_current_weather(city: str) -> str:
        """get current weather"""
        return "Not implemented"

    agent = create_agent(
        model="dashscope:qwen-flash",
        tools=[get_current_weather],
        middleware=[middleware],
    )
    response = agent.invoke(
        {"messages": [HumanMessage("What's the weather in New York?")]}
    )

    message = response["messages"][-2]
    assert isinstance(message, ToolMessage)
    assert message.content != "Not implemented"


@pytest.mark.asyncio
async def test_model_tool_emulator_async():
    middleware = LLMToolEmulator(model="dashscope:qwen-flash")

    @tool
    def get_current_weather(city: str) -> str:
        """get current weather"""
        return "Not implemented"

    agent = create_agent(
        model="dashscope:qwen-flash",
        tools=[get_current_weather],
        middleware=[middleware],
    )
    response = await agent.ainvoke(
        {"messages": [HumanMessage("What's the weather in New York?")]}
    )

    message = response["messages"][-2]
    assert isinstance(message, ToolMessage)
    assert message.content != "Not implemented"
