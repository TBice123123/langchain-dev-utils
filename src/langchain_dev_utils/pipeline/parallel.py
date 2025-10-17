from typing import Awaitable, Callable, Optional, Union

from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send
from langgraph.typing import ContextT, InputT, OutputT, StateT

from .types import SubGraph


def parallel_pipeline(
    sub_graphs: list[SubGraph],
    state_schema: type[StateT],
    graph_name: Optional[str] = None,
    branches_fn: Optional[
        Union[
            Callable[..., list[Send]],
            Callable[..., Awaitable[list[Send]]],
        ]
    ] = None,
    context_schema: type[ContextT] | None = None,
    input_schema: type[InputT] | None = None,
    output_schema: type[OutputT] | None = None,
) -> CompiledStateGraph[StateT, ContextT, InputT, OutputT]:
    """
    Create a parallel pipeline from a list of subgraphs.

    This function allows you to compose multiple StateGraphs in a parallel fashion,
    where subgraphs can execute concurrently. This is useful for creating complex
    multi-agent workflows where agents can work independently or with dynamic branching.

    Args:
        sub_graphs: List of sub-graphs to execute in parallel
        state_schema: state schema of the final constructed graph
        graph_name: Name of the final constructed graph
        branches_fn: Optional function to determine which sub-graphs to execute in parallel
        context_schema: context schema of the final constructed graph
        input_schema: input schema of the final constructed graph
        output_schema: output schema of the final constructed graph

    Returns:
        CompiledStateGraph[StateT, ContextT, InputT, OutputT]: Compiled state graph of the pipeline.

    Example:
        Parallel pipeline with entry graph:
        >>> from langchain_dev_utils.pipeline import parallel_pipeline
        >>> from src.graph import graph1,graph2
        >>> from src.state import State
        >>>
        >>> graph = parallel_pipeline(
        ...     sub_graphs=[graph1,graph2],
        ...     state_schema=State,
        ...     graph_name="parallel graph",
        ... )
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

    if branches_fn:
        graph.add_conditional_edges(
            "__start__",
            branches_fn,
            [subgraph.name for subgraph in compiled_subgraphs],
        )
        return graph.compile(name=graph_name or "parallel graph")
    else:
        for i in range(len(compiled_subgraphs)):
            graph.add_edge("__start__", compiled_subgraphs[i].name)
        return graph.compile(name=graph_name or "parallel graph")
