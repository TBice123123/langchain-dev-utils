from typing import Optional, Callable
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send
from langgraph.typing import StateT, ContextT, InputT, OutputT
from langchain_dev_utils.graph_pipeline.types import SubGraph


def parallel_pipeline(
    sub_graphs: list[SubGraph],
    state_schema: type[StateT],
    graph_name: Optional[str] = None,
    parallel_entry_graph: Optional[str] = None,
    branches_fn: Optional[Callable[[StateT], list[Send]]] = None,
    context_schema: type[ContextT] | None = None,
    input_schema: type[InputT] | None = None,
    output_schema: type[OutputT] | None = None,
) -> CompiledStateGraph[StateT, ContextT, InputT, OutputT]:
    """
    Create a parallel pipeline from a list of subgraphs.

    Args:
        sub_graphs: List of sub-graphs
        state_schema: state schema of the final constructed graph
        graph_name: Name of the final constructed graph
        parallel_entry_graph: Parallel entry graph
        branches_fn: Function to determine which sub-graphs to execute in parallel
        context_schema: context schema of the final constructed graph
        input_schema: input schema of the final constructed graph
        output_schema: output schema of the final constructed graph

    Returns:
        CompiledStateGraph[StateT, ContextT, InputT, OutputT]: Compiled state graph of the pipeline.
    """
    graph = StateGraph(
        state_schema=state_schema,
        context_schema=context_schema,
        input_schema=input_schema,
        output_schema=output_schema,
    )

    subgraphs_names = set()

    compiled_subgraphs: list[CompiledStateGraph] = []
    for subgraph in sub_graphs:
        if isinstance(subgraph, StateGraph):
            subgraph = subgraph.compile()

        compiled_subgraphs.append(subgraph)
        if subgraph.name is None or subgraph.name == "LangGraph":
            raise ValueError(
                "Please specify a name when you create your agent, either via `create_react_agent(..., name=agent_name)` "
                "or via `graph.compile(name=name)`."
            )

        if subgraph.name in subgraphs_names:
            raise ValueError(
                f"Subgraph with name '{subgraph.name}' already exists. Subgraph names must be unique."
            )

        subgraphs_names.add(subgraph.name)

    for sub_graph in compiled_subgraphs:
        graph.add_node(sub_graph.name, sub_graph)

    if parallel_entry_graph and parallel_entry_graph not in subgraphs_names:
        raise ValueError(
            f"Parallel entry graph '{parallel_entry_graph}' does not exist."
        )

    entry_graph = parallel_entry_graph or "__start__"

    if entry_graph != "__start__":
        graph.add_edge("__start__", entry_graph)

    if branches_fn:
        graph.add_conditional_edges(
            entry_graph,
            branches_fn,
            [
                subgraph.name
                for subgraph in compiled_subgraphs
                if subgraph.name != entry_graph
            ],
        )
        return graph.compile(name=graph_name or "parallel graph")
    else:
        filtered_subgraphs = [
            subgraph for subgraph in compiled_subgraphs if subgraph.name != entry_graph
        ]
        for i in range(len(filtered_subgraphs)):
            graph.add_edge(entry_graph, filtered_subgraphs[i].name)
        return graph.compile(name=graph_name or "parallel graph")
