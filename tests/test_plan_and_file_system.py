import pytest
from langgraph.prebuilt.tool_node import ToolNode
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.message import MessagesState
from langgraph.graph.state import StateGraph

from langchain_dev_utils.agents.file_system import (
    FileStateMixin,
    create_ls_file_tool,
    create_query_file_tool,
    create_update_file_tool,
    create_write_file_tool,
)
from langchain_dev_utils.agents.plan import (
    PlanStateMixin,
    create_update_plan_tool,
    create_write_plan_tool,
)


def build_graph():
    class State(MessagesState, PlanStateMixin, FileStateMixin):
        pass

    class StateIn(MessagesState):
        pass

    def make_plan(state: State):
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "write_plan",
                            "args": {"plan": ["plan1", "plan2"]},
                            "id": "123",
                        }
                    ],
                )
            ]
        }

    def test_make_plan(state: State):
        assert state["plan"] == [
            {"content": "plan1", "status": "in_progress"},
            {"content": "plan2", "status": "pending"},
        ]
        return {"messages": [HumanMessage(content="")]}

    def update_plan(state: State):
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "update_plan",
                            "args": {
                                "update_plans": [
                                    {"content": "plan1", "status": "done"},
                                    {"content": "plan2", "status": "in_progress"},
                                ]
                            },
                            "id": "1234",
                        }
                    ],
                )
            ]
        }

    def test_update_plan(state: State):
        assert state["plan"] == [
            {"content": "plan1", "status": "done"},
            {"content": "plan2", "status": "in_progress"},
        ]
        return {"messages": [HumanMessage(content="")]}

    def write_file1(state: State):
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "write_file",
                            "args": {"file_name": "file1", "content": "content1"},
                            "id": "12345",
                        },
                        {
                            "name": "write_file",
                            "args": {"file_name": "file2", "content": "content2"},
                            "id": "123456",
                        },
                    ],
                )
            ]
        }

    def write_file2(state: State):
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "write_file",
                            "args": {
                                "file_name": "file1",
                                "content": "\ncontent1",
                                "write_mode": "append",
                            },
                            "id": "1234567",
                        },
                    ],
                )
            ]
        }

    def test_write_file(state: State):
        assert state["file"] == {
            "file1": "content1\ncontent1",
            "file2": "content2",
        }
        return {}

    def update_file(state: State):
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "update_file",
                            "args": {
                                "file_name": "file1",
                                "origin_content": "content1",
                                "new_content": "content_new",
                                "replace_all": True,
                            },
                            "id": "12345",
                        },
                        {
                            "name": "update_file",
                            "args": {
                                "file_name": "file2",
                                "origin_content": "content2",
                                "new_content": "content_new2",
                            },
                            "id": "123456",
                        },
                    ],
                )
            ]
        }

    def test_update_file(state: State):
        assert state["file"] == {
            "file1": "content_new\ncontent_new",
            "file2": "content_new2",
        }
        return {}

    graph = StateGraph(State, input_schema=StateIn)
    graph.add_node("make_plan", make_plan)
    graph.add_node("test_make_plan", test_make_plan)
    graph.add_node("update_plan", update_plan)
    graph.add_node("test_update_plan", test_update_plan)
    graph.add_node("write_file1", write_file1)
    graph.add_node("write_file2", write_file2)
    graph.add_node("test_write_file", test_write_file)
    graph.add_node("update_file", update_file)
    graph.add_node("test_update_file", test_update_file)
    graph.add_node(
        "write_plan_tool_node",
        ToolNode(
            [
                create_write_plan_tool(),
            ]
        ),
    )
    graph.add_node(
        "update_plan_tool_node",
        ToolNode(
            [
                create_update_plan_tool(),
            ]
        ),
    )
    graph.add_node(
        "write_file_tool_node1",
        ToolNode([create_write_file_tool()]),
    )
    graph.add_node(
        "write_file_tool_node2",
        ToolNode([create_write_file_tool()]),
    )

    graph.add_node(
        "update_file_tool_node",
        ToolNode([create_update_file_tool()]),
    )

    graph.set_entry_point("make_plan")
    graph.add_edge("make_plan", "write_plan_tool_node")
    graph.add_edge("write_plan_tool_node", "test_make_plan")
    graph.add_edge("test_make_plan", "update_plan")
    graph.add_edge("update_plan", "update_plan_tool_node")
    graph.add_edge("update_plan_tool_node", "test_update_plan")
    graph.add_edge("test_update_plan", "write_file1")
    graph.add_edge("write_file1", "write_file_tool_node1")
    graph.add_edge("write_file_tool_node1", "write_file2")
    graph.add_edge("write_file2", "write_file_tool_node2")
    graph.add_edge("write_file_tool_node2", "test_write_file")
    graph.add_edge("test_write_file", "update_file")
    graph.add_edge("update_file", "update_file_tool_node")
    graph.add_edge("update_file_tool_node", "test_update_file")
    return graph


def test_plan_tool():
    write_plan_tool = create_write_plan_tool()
    update_plan_tool = create_update_plan_tool()

    assert write_plan_tool.name == "write_plan"
    assert update_plan_tool.name == "update_plan"

    write_file_tool_with_name = create_write_file_tool(name="write")
    update_file_tool_with_name = create_update_file_tool(name="update")
    assert write_file_tool_with_name.name == "write"
    assert update_file_tool_with_name.name == "update"

    write_file_tool_with_description = create_write_file_tool(description="write")
    update_file_tool_with_description = create_update_file_tool(description="update")
    assert write_file_tool_with_description.description == "write"
    assert update_file_tool_with_description.description == "update"

    write_file_tool_with_name_and_description = create_write_file_tool(
        name="write_tool", description="write"
    )
    update_file_tool_with_name_and_description = create_update_file_tool(
        name="update_tool", description="update"
    )
    assert write_file_tool_with_name_and_description.name == "write_tool"
    assert update_file_tool_with_name_and_description.name == "update_tool"
    assert write_file_tool_with_name_and_description.description == "write"
    assert update_file_tool_with_name_and_description.description == "update"


def test_file_tool():
    write_file_tool = create_write_file_tool()
    ls_tool = create_ls_file_tool()
    query_file_tool = create_query_file_tool()
    assert write_file_tool.name == "write_file"
    assert ls_tool.name == "ls"
    assert query_file_tool.name == "query_file"

    write_file_tool_with_name = create_write_file_tool(name="write")
    ls_tool_with_name = create_ls_file_tool(name="list")
    query_file_tool_with_name = create_query_file_tool(name="query")
    assert write_file_tool_with_name.name == "write"
    assert ls_tool_with_name.name == "list"
    assert query_file_tool_with_name.name == "query"

    write_file_tool_with_description = create_write_file_tool(description="write")
    ls_tool_with_description = create_ls_file_tool(description="list")
    query_file_tool_with_description = create_query_file_tool(description="query")
    assert write_file_tool_with_description.description == "write"
    assert ls_tool_with_description.description == "list"
    assert query_file_tool_with_description.description == "query"

    write_file_tool_with_name_and_description = create_write_file_tool(
        name="write_file_tool", description="write"
    )
    ls_tool_with_name_and_description = create_ls_file_tool(
        name="list", description="list files"
    )
    query_file_tool_with_name_and_description = create_query_file_tool(
        name="query_file_tool", description="query"
    )
    assert write_file_tool_with_name_and_description.name == "write_file_tool"
    assert ls_tool_with_name_and_description.name == "list"
    assert query_file_tool_with_name_and_description.name == "query_file_tool"
    assert write_file_tool_with_name_and_description.description == "write"
    assert ls_tool_with_name_and_description.description == "list files"
    assert query_file_tool_with_name_and_description.description == "query"


@pytest.mark.asyncio
async def test_invoke():
    graph = build_graph()
    graph = graph.compile()
    await graph.ainvoke({"messages": [HumanMessage(content="")]})
