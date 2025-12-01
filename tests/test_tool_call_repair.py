from typing import Any, Callable, Sequence

from langchain.agents import create_agent
from langchain.tools import BaseTool, tool
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.fake_chat_models import FakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from typing_extensions import override

from langchain_dev_utils.agents.middleware import ToolCallRepairMiddleware


class MockChatModel(FakeChatModel):
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if isinstance(messages[-1], HumanMessage):
            message = AIMessage(
                content="",
                invalid_tool_calls=[
                    {
                        "id": "call_123",
                        "name": "get_weather",
                        "args": "{'city': 'New York'}}",
                        "error": "Invalid email address",
                        "type": "invalid_tool_call",
                    }
                ],
            )
            generation = ChatGeneration(message=message)
        else:
            generation = ChatGeneration(message=AIMessage(content=messages[-1].content))
        return ChatResult(generations=[generation])

    @override
    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if isinstance(messages[-1], HumanMessage):
            message = AIMessage(
                content="",
                invalid_tool_calls=[
                    {
                        "id": "call_123",
                        "name": "get_weather",
                        "args": "{'city': 'New York'}}",
                        "error": "Invalid email address",
                        "type": "invalid_tool_call",
                    }
                ],
            )
            generation = ChatGeneration(message=message)
        else:
            generation = ChatGeneration(
                message=AIMessage(content="New York has a sunny weather")
            )
        return ChatResult(generations=[generation])

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ):
        return super().bind(**kwargs)


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."


def test_tool_call_repair():
    model = MockChatModel()
    agent = create_agent(
        model,
        tools=[get_weather],
        middleware=[ToolCallRepairMiddleware()],
    )
    result = agent.invoke({"messages": [HumanMessage(content="New York Weather?")]})

    ai_message = result["messages"][1]
    assert (
        isinstance(ai_message, AIMessage)
        and len(ai_message.tool_calls) > 0
        and len(ai_message.invalid_tool_calls) == 0
    )


async def test_tool_call_repair_async():
    model = MockChatModel()
    agent = create_agent(
        model,
        tools=[get_weather],
        middleware=[ToolCallRepairMiddleware()],
    )
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="New York Weather?")]}
    )

    ai_message = result["messages"][1]
    assert (
        isinstance(ai_message, AIMessage)
        and len(ai_message.tool_calls) > 0
        and len(ai_message.invalid_tool_calls) == 0
    )


def test_tool_call_repair_with_no_invalid_tool_call():
    agent = create_agent(
        "deepseek-chat",
        tools=[get_weather],
        middleware=[ToolCallRepairMiddleware()],
    )
    result = agent.invoke({"messages": [HumanMessage(content="New York Weather?")]})

    ai_message = result["messages"][1]
    assert (
        isinstance(ai_message, AIMessage)
        and len(ai_message.tool_calls) > 0
        and len(ai_message.invalid_tool_calls) == 0
    )


async def test_tool_call_repair_with_no_invalid_tool_call_async():
    agent = create_agent(
        "deepseek-chat",
        tools=[get_weather],
        middleware=[ToolCallRepairMiddleware()],
    )
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="New York Weather?")]}
    )

    ai_message = result["messages"][1]
    assert (
        isinstance(ai_message, AIMessage)
        and len(ai_message.tool_calls) > 0
        and len(ai_message.invalid_tool_calls) == 0
    )
