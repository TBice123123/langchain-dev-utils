import datetime
from typing import Annotated, Any
from typing import cast

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.graph.message import add_messages
from langgraph.graph.state import StateGraph
from langgraph.types import interrupt
import pytest
from typing_extensions import TypedDict

from langchain_dev_utils.tool_calling import human_in_the_loop, human_in_the_loop_async
from langchain_dev_utils.tool_calling.human_in_loop import InterruptParams


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
    class State(TypedDict):
        timestamp: str
        messages: Annotated[list[BaseMessage], add_messages]

    def run_tool(state: State) -> State:
        timestamp = tool.invoke({})
        return {"timestamp": timestamp, "messages": state["messages"]}

    graph1 = StateGraph(State)
    graph1.add_node("tool", run_tool)
    graph1.add_edge("__start__", "tool")

    graph1 = graph1.compile()

    for msg in graph1.stream(
        {
            "timestamp": "",
            "messages": [HumanMessage("1")],
        }
    ):
        assert "__interrupt__" in msg
        assert cast(tuple, msg.get("__interrupt__"))[0].value == expected


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
    class State(TypedDict):
        timestamp: str
        messages: Annotated[list[BaseMessage], add_messages]

    async def run_tool(state: State) -> State:
        timestamp = await tool.ainvoke({})
        return {"timestamp": timestamp, "messages": state["messages"]}

    graph = StateGraph(State)
    graph.add_node("tool", run_tool)
    graph.add_edge("__start__", "tool")

    graph = graph.compile()

    async for msg in graph.astream(
        {
            "timestamp": "",
            "messages": [HumanMessage("1")],
        }
    ):
        assert "__interrupt__" in msg
        assert cast(tuple, msg.get("__interrupt__"))[0].value == expected
