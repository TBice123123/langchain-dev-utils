import pytest
from dotenv import load_dotenv
from langchain.agents.structured_output import ToolStrategy
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from langchain_dev_utils.agents import create_agent
from langchain_dev_utils.chat_models import batch_register_model_provider

load_dotenv()

batch_register_model_provider(
    [
        {
            "provider": "dashscope",
            "chat_model": ChatTongyi,
        },
        {
            "provider": "zai",
            "chat_model": "openai-compatible",
            "base_url": "https://open.bigmodel.cn/api/paas/v4",
        },
    ]
)


def test_prebuilt_agent():
    from langchain_core.tools import tool

    @tool
    def get_current_weather(city: str) -> str:
        """get current weather"""
        return f"in {city}, it is sunny"

    agent = create_agent(model="dashscope:qwen-flash", tools=[get_current_weather])
    response = agent.invoke(
        {"messages": [HumanMessage("What's the weather in New York?")]}
    )
    assert len(response["messages"]) == 4

    assert response["messages"][1].tool_calls[0]["name"] == "get_current_weather"


@pytest.mark.asyncio
async def test_prebuilt_agent_async():
    from langchain_core.tools import tool

    @tool
    def get_current_weather(city: str) -> str:
        """get current weather"""
        return f"in {city}, it is sunny"

    agent = create_agent(model="dashscope:qwen-flash", tools=[get_current_weather])
    response = await agent.ainvoke(
        {"messages": [HumanMessage("What's the weather in New York?")]}
    )
    assert len(response["messages"]) == 4

    assert response["messages"][1].tool_calls[0]["name"] == "get_current_weather"


class WeatherBaseModel(BaseModel):
    """Weather response."""

    temperature: float = Field(description="The temperature in fahrenheit")
    condition: str = Field(description="Weather condition")


def get_weather(city: str) -> str:  # noqa: ARG001
    """Get the weather for a city."""
    return "The weather is sunny and 75°F."


def test_inference_to_tool_output():
    agent = create_agent(
        model="zai:glm-4.6",
        system_prompt=(
            "You are a helpful weather assistant. Please call the get_weather tool, "
            "then use the **WeatherBaseModel** to generate the final response."
        ),
        tools=[get_weather],
        response_format=ToolStrategy(WeatherBaseModel),
    )
    response = agent.invoke(
        {"messages": [HumanMessage("What's the weather in New York? ")]}
    )

    assert isinstance(response["structured_response"], WeatherBaseModel)
    assert response["structured_response"].temperature == 75.0
    assert response["structured_response"].condition.lower() == "sunny"
    assert len(response["messages"]) == 5

    assert [m.type for m in response["messages"]] == [
        "human",
        "ai",
        "tool",
        "ai",
        "tool",
    ]
