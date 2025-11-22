import datetime
from typing import Any
from typing import cast

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langchain_qwq import ChatQwen
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt
import pytest

from langchain_dev_utils.chat_models import load_chat_model, register_model_provider
from langchain_dev_utils.tool_calling import (
    InterruptParams,
    human_in_the_loop,
    human_in_the_loop_async,
)

load_dotenv()
register_model_provider(provider_name="dashscope", chat_model=ChatQwen)


def handler(params: InterruptParams):
    response = interrupt(f"Please review tool call: {params['tool_call_name']}")
    if response["type"] == "accept":
        return params["tool"].invoke(params["tool_call_args"])
    elif response["type"] == "edit":
        updated_args = response["args"]["args"]
        return params["tool"].invoke(updated_args)
    elif response["type"] == "response":
        return response["args"]
    else:
        raise ValueError(f"Unsupported interrupt response type: {response['type']}")


@human_in_the_loop
def get_current_time() -> str:
    """Get current timestamp"""
    return str(datetime.datetime.now().timestamp())


@human_in_the_loop_async
async def get_current_time_async() -> str:
    """Get current timestamp"""
    return str(datetime.datetime.now().timestamp())


@human_in_the_loop(handler=handler)
def get_current_time_with_handler() -> str:
    """Get current timestamp"""
    return str(datetime.datetime.now().timestamp())


@human_in_the_loop_async(handler=handler)
async def get_current_time_with_handler_async() -> str:
    """Get current timestamp"""
    return str(datetime.datetime.now().timestamp())


@pytest.mark.parametrize(
    "tool,expected",
    [
        (
            get_current_time,
            {
                "action_request": {"action": "get_current_time", "args": {}},
                "config": {
                    "allow_accept": True,
                    "allow_edit": True,
                    "allow_respond": True,
                },
                "description": "Please review tool call: get_current_time",
            },
        ),
        (
            get_current_time_with_handler,
            "Please review tool call: get_current_time_with_handler",
        ),
    ],
)
def test_human_in_loop(tool: BaseTool, expected: Any):
    model = load_chat_model("dashscope:qwen-flash")

    agent = create_agent(
        model=model,
        tools=[tool],
        checkpointer=InMemorySaver(),
    )

    response = agent.invoke(
        {"messages": [HumanMessage("what's the time")]},
        config={"configurable": {"thread_id": "1"}},
    )
    assert "__interrupt__" in response
    assert cast(tuple, response.get("__interrupt__"))[0].value == expected


@pytest.mark.parametrize(
    "tool,expected",
    [
        (
            get_current_time_async,
            {
                "action_request": {"action": "get_current_time_async", "args": {}},
                "config": {
                    "allow_accept": True,
                    "allow_edit": True,
                    "allow_respond": True,
                },
                "description": "Please review tool call: get_current_time_async",
            },
        ),
        (
            get_current_time_with_handler_async,
            "Please review tool call: get_current_time_with_handler_async",
        ),
    ],
)
async def test_human_in_loop_async(tool: BaseTool, expected: Any):
    model = load_chat_model("dashscope:qwen-flash")

    agent = create_agent(
        model=model,
        tools=[tool],
        checkpointer=InMemorySaver(),
    )

    response = await agent.ainvoke(
        {"messages": [HumanMessage("what's the time")]},
        config={"configurable": {"thread_id": "1"}},
    )
    assert "__interrupt__" in response
    assert cast(tuple, response.get("__interrupt__"))[0].value == expected
