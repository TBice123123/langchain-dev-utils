from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_core.language_models import BaseChatModel
import pytest

from langchain_dev_utils.chat_models import (
    batch_register_model_provider,
    load_chat_model,
)

load_dotenv()

batch_register_model_provider(
    [
        {
            "provider_name": "dashscope",
            "chat_model": ChatTongyi,
        },
        {"provider_name": "zai", "chat_model": "openai-compatible"},
    ]
)


@pytest.fixture(
    params=["dashscope:qwen-flash", "zai:glm-4.6", "deepseek:deepseek-chat"]
)
def model(request: pytest.FixtureRequest):
    params = request.param
    if params == "zai:glm-4.6":
        return load_chat_model(
            params,
            extra_body={
                "thinking": {
                    "type": "disabled",
                }
            },
        )
    return load_chat_model(params)


@pytest.fixture
def reasoning_model():
    return load_chat_model("zai:glm-4.6")


def test_model_invoke(
    model: BaseChatModel,
):
    response = model.invoke("what's your name")
    assert isinstance(response.content, str)


@pytest.mark.asyncio
async def test_model_ainvoke(
    model: BaseChatModel,
):
    response = await model.ainvoke("what's your name")
    assert isinstance(response.content, str)


def test_model_tool_calling(
    model: BaseChatModel,
):
    from langchain_core.tools import tool

    @tool
    def get_current_weather(city: str) -> str:
        """get current weather"""
        return f"in {city}, it is sunny"

    bind_model = model.bind_tools([get_current_weather])

    response = bind_model.invoke("what's the weather in new york")
    assert hasattr(response, "tool_calls") and len(response.tool_calls) == 1


@pytest.mark.asyncio
async def test_model_tool_calling_async(
    model: BaseChatModel,
):
    from langchain_core.tools import tool

    @tool
    def get_current_weather(city: str) -> str:
        """get current weather"""
        return f"in {city}, it is sunny"

    bind_model = model.bind_tools([get_current_weather])

    response = await bind_model.ainvoke("what's the weather in new york")
    assert hasattr(response, "tool_calls") and len(response.tool_calls) == 1


def test_model_with_reasoning(reasoning_model: BaseChatModel):
    response = reasoning_model.invoke("hello?")
    assert response.additional_kwargs.get("reasoning_content")


@pytest.mark.asyncio
async def test_model_with_reasoning_async(reasoning_model: BaseChatModel):
    response = await reasoning_model.ainvoke("hello?")
    assert response.additional_kwargs.get("reasoning_content")
