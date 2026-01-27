from typing import Annotated, TypedDict

from langgraph.types import Send

from langchain_dev_utils.graph import (
    create_parallel_graph,
    create_sequential_graph,
)


def replace(a: int, b: int):
    return b


class State(TypedDict):
    a: Annotated[int, replace]


def branches_fn(state: State):
    return [
        Send("add1", arg={"a": state["a"] + 1}),
        Send("add2", arg={"a": state["a"] + 1}),
    ]


def add1(state: State):
    return {"a": state["a"] + 1}


def add2(state: State):
    return {"a": state["a"] + 1}


def add3(state: State):
    return {"a": state["a"] + 1}


def test_sequential_graph():
    graph = create_sequential_graph(
        nodes=[add1, add2, add3],
        state_schema=State,
    )
    result = graph.invoke({"a": 1})
    assert result["a"] == 4


def test_parallel_graph():
    graph = create_parallel_graph(
        nodes=[add1, add2, add3],
        state_schema=State,
    )
    result = graph.invoke({"a": 1})
    assert result["a"] == 2


def test_parallel_graph_with_branches_fn():
    graph = create_parallel_graph(
        nodes=[add1, add2, add3],
        state_schema=State,
        branches_fn=branches_fn,
    )

    result = graph.invoke({"a": 1})
    assert result["a"] == 3
