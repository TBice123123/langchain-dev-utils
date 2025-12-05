import pytest
from langchain_core.language_models import BaseChatModel

from data.alibaba._profiles import _PROFILES as ALI_PROFILES
from data.zhipuai._profiles import _PROFILES as ZAI_PROFILES
from langchain_dev_utils.chat_models import load_chat_model


@pytest.fixture(
    params=["dashscope:qwen-flash", "zai:glm-4.6", "deepseek:deepseek-reasoner"]
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
    response = await reasoning_model.ainvoke("hello？")
    assert response.additional_kwargs.get("reasoning_content")


def test_model_profile():
    model = load_chat_model("dashscope:qwen-flash")
    assert model.profile == ALI_PROFILES["qwen-flash"]

    model = load_chat_model("dashscope:qwen-max")
    assert model.profile == ALI_PROFILES["qwen-max"]

    model = load_chat_model("dashscope:qwen3-vl-235b-a22b")
    assert model.profile == ALI_PROFILES["qwen3-vl-235b-a22b"]

    model = load_chat_model("zai:glm-4.6")
    assert model.profile == ZAI_PROFILES["glm-4.6"]

    model = load_chat_model("zai:glm-4.5v")
    assert model.profile == ZAI_PROFILES["glm-4.5v"]
