from langchain.schema import HumanMessage
from langchain_core.messages import AIMessage
from langgraph.graph.message import MessagesState
from langgraph.graph.state import StateGraph
import pytest
from langchain_dev_utils import (
    NoteStateMixin,
    PlanStateMixin,
    create_query_note_tool,
    create_write_plan_tool,
    create_write_note_tool,
    create_update_plan_tool,
    create_ls_tool,
    create_update_note_tool,
)

from langgraph.prebuilt.tool_node import ToolNode


def build_graph():
    class State(MessagesState, PlanStateMixin, NoteStateMixin):
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

    def write_note(state: State):
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "write_note",
                            "args": {"file_name": "note1", "content": "note1 content"},
                            "id": "12345",
                        },
                        {
                            "name": "write_note",
                            "args": {"file_name": "note2", "content": "note2 content"},
                            "id": "123456",
                        },
                    ],
                )
            ]
        }

    def test_write_note(state: State):
        assert state["note"] == {
            "note1": "note1 content",
            "note2": "note2 content",
        }
        return {}

    def update_note(state: State):
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "update_note",
                            "args": {
                                "file_name": "note1",
                                "origin_content": "note1",
                                "new_content": "note_new",
                            },
                            "id": "12345",
                        },
                    ],
                )
            ]
        }

    def test_update_note(state: State):
        assert state["note"] == {
            "note1": "note_new content",
            "note2": "note2 content",
        }
        return {}

    graph = StateGraph(State, input_schema=StateIn)
    graph.add_node("make_plan", make_plan)
    graph.add_node("test_make_plan", test_make_plan)
    graph.add_node("update_plan", update_plan)
    graph.add_node("test_update_plan", test_update_plan)
    graph.add_node("write_note", write_note)
    graph.add_node("test_write_note", test_write_note)
    graph.add_node("update_note", update_note)
    graph.add_node("test_update_note", test_update_note)
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
        "write_note_tool_node",
        ToolNode([create_write_note_tool()]),
    )

    graph.add_node(
        "update_note_tool_node",
        ToolNode([create_update_note_tool()]),
    )

    graph.set_entry_point("make_plan")
    graph.add_edge("make_plan", "write_plan_tool_node")
    graph.add_edge("write_plan_tool_node", "test_make_plan")
    graph.add_edge("test_make_plan", "update_plan")
    graph.add_edge("update_plan", "update_plan_tool_node")
    graph.add_edge("update_plan_tool_node", "test_update_plan")
    graph.add_edge("test_update_plan", "write_note")
    graph.add_edge("write_note", "write_note_tool_node")
    graph.add_edge("write_note_tool_node", "test_write_note")
    graph.add_edge("test_write_note", "update_note")
    graph.add_edge("update_note", "update_note_tool_node")
    graph.add_edge("update_note_tool_node", "test_update_note")
    return graph


def test_plan_tool():
    write_plan_tool = create_write_plan_tool()
    update_plan_tool = create_update_plan_tool()

    assert write_plan_tool.name == "write_plan"
    assert update_plan_tool.name == "update_plan"

    write_note_tool_with_name = create_write_note_tool(name="write")
    update_note_tool_with_name = create_update_plan_tool(name="update")
    assert write_note_tool_with_name.name == "write"
    assert update_note_tool_with_name.name == "update"

    write_note_tool_with_description = create_write_note_tool(description="write")
    update_note_tool_with_description = create_update_plan_tool(description="update")
    assert write_note_tool_with_description.description == "write"
    assert update_note_tool_with_description.description == "update"

    write_note_tool_with_name_and_description = create_write_note_tool(
        name="write_tool", description="write"
    )
    update_note_tool_with_name_and_description = create_update_plan_tool(
        name="update_tool", description="update"
    )
    assert write_note_tool_with_name_and_description.name == "write_tool"
    assert update_note_tool_with_name_and_description.name == "update_tool"
    assert write_note_tool_with_name_and_description.description == "write"
    assert update_note_tool_with_name_and_description.description == "update"


def test_note_tool():
    write_note_tool = create_write_note_tool()
    ls_tool = create_ls_tool()
    query_note_tool = create_query_note_tool()
    assert write_note_tool.name == "write_note"
    assert ls_tool.name == "ls"
    assert query_note_tool.name == "query_note"

    write_note_tool_with_name = create_write_note_tool(name="write")
    ls_tool_with_name = create_ls_tool(name="list")
    query_note_tool_with_name = create_query_note_tool(name="query")
    assert write_note_tool_with_name.name == "write"
    assert ls_tool_with_name.name == "list"
    assert query_note_tool_with_name.name == "query"

    write_note_tool_with_description = create_write_note_tool(description="write")
    ls_tool_with_description = create_ls_tool(description="list")
    query_note_tool_with_description = create_query_note_tool(description="query")
    assert write_note_tool_with_description.description == "write"
    assert ls_tool_with_description.description == "list"
    assert query_note_tool_with_description.description == "query"

    write_note_tool_with_name_and_description = create_write_note_tool(
        name="write_note_tool", description="write"
    )
    ls_tool_with_name_and_description = create_ls_tool(
        name="list", description="list notes"
    )
    query_note_tool_with_name_and_description = create_query_note_tool(
        name="query_note_tool", description="query"
    )
    assert write_note_tool_with_name_and_description.name == "write_note_tool"
    assert ls_tool_with_name_and_description.name == "list"
    assert query_note_tool_with_name_and_description.name == "query_note_tool"
    assert write_note_tool_with_name_and_description.description == "write"
    assert ls_tool_with_name_and_description.description == "list notes"
    assert query_note_tool_with_name_and_description.description == "query"


@pytest.mark.asyncio
async def test_invoke():
    graph = build_graph()
    graph = graph.compile()
    await graph.ainvoke({"messages": [HumanMessage(content="")]})
