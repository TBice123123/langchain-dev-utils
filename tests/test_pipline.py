from typing import Annotated, TypedDict

from langgraph.graph import StateGraph
from langgraph.types import Send

from langchain_dev_utils.pipeline import (
    create_parallel_pipeline,
    create_sequential_pipeline,
)


def replace(a: int, b: int):
    return b


class State(TypedDict):
    a: Annotated[int, replace]


def branches_fn(state: State):
    return [
        Send("graph1", arg={"a": state["a"]}),
        Send("graph2", arg={"a": state["a"]}),
    ]


def add(state: State):
    return {"a": state["a"] + 1}


def make_graph(name: str):
    sub_graph = StateGraph(State)
    sub_graph.add_node("add", add)
    sub_graph.add_edge("__start__", "add")
    return sub_graph.compile(name=name)


def test_sequential_graph():
    graph = create_sequential_pipeline(
        sub_graphs=[
            make_graph("graph1"),
            make_graph("graph2"),
            make_graph("graph3"),
        ],
        state_schema=State,
    )
    result = graph.invoke({"a": 1})
    assert result["a"] == 4


def test_parallel_graph():
    graph = create_parallel_pipeline(
        sub_graphs=[
            make_graph("graph1"),
            make_graph("graph2"),
            make_graph("graph3"),
        ],
        state_schema=State,
    )
    result = graph.invoke({"a": 1})
    assert result["a"] == 2


def test_parallel_graph_with_branches_fn():
    graph = create_parallel_pipeline(
        sub_graphs=[
            make_graph("graph1"),
            make_graph("graph2"),
            make_graph("graph3"),
        ],
        state_schema=State,
        branches_fn=branches_fn,
    )

    result = graph.invoke({"a": 1})
    assert result["a"] == 2
