from langgraph.graph import StateGraph
from typing import Annotated, TypedDict
from langchain_dev_utils import sequential_pipeline, parallel_pipeline


def replace(a: int, b: int):
    return b


class State(TypedDict):
    a: Annotated[int, replace]


def add(state: State):
    print(state)
    return {"a": state["a"] + 1}


def make_graph(name: str):
    sub_graph = StateGraph(State)
    sub_graph.add_node("add", add)
    sub_graph.add_edge("__start__", "add")
    return sub_graph.compile(name=name)


def test_sequential_graph():
    graph = sequential_pipeline(
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
    graph = parallel_pipeline(
        sub_graphs=[
            make_graph("graph1"),
            make_graph("graph2"),
            make_graph("graph3"),
        ],
        state_schema=State,
    )
    result = graph.invoke({"a": 1})
    assert result["a"] == 2


def test_parallel_graph_with_entry_note():
    graph = parallel_pipeline(
        sub_graphs=[
            make_graph("graph1"),
            make_graph("graph2"),
            make_graph("graph3"),
        ],
        state_schema=State,
        parallel_entry_node="graph1",
    )

    result = graph.invoke({"a": 1})
    assert result["a"] == 3
