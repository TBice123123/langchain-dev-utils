from typing import Optional

from langgraph.cache.base import BaseCache
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer
from langgraph.typing import ContextT, InputT, OutputT, StateT

from .types import SubGraph


def create_sequential_pipeline(
    sub_graphs: list[SubGraph],
    state_schema: type[StateT],
    graph_name: Optional[str] = None,
    context_schema: type[ContextT] | None = None,
    input_schema: type[InputT] | None = None,
    output_schema: type[OutputT] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    cache: BaseCache | None = None,
) -> CompiledStateGraph[StateT, ContextT, InputT, OutputT]:
    """
    Create a sequential pipeline from a list of subgraphs.

    This function allows you to compose multiple StateGraphs in a sequential fashion,
    where each subgraph executes one after another. This is useful for creating
    complex multi-agent workflows where agents need to work in a specific order.

    Args:
        sub_graphs: List of sub-graphs to execute sequentially
        state_schema: state schema of the final constructed graph
        graph_name: Name of the final constructed graph
        context_schema: context schema of the final constructed graph
        input_schema: input schema of the final constructed graph
        output_schema: output schema of the final constructed graph
        checkpointer: Optional LangGraph checkpointer for the final constructed
            graph
        store: Optional LangGraph store for the final constructed graph
        cache: Optional LangGraph cache for the final constructed graph

    Returns:
        CompiledStateGraph[StateT, ContextT, InputT, OutputT]: Compiled state
            graph of the pipeline.

    Example:
        # Basic sequential pipeline with multiple specialized agents:
        >>> from langchain_dev_utils.pipeline import create_sequential_pipeline
        >>>
        >>> graph = create_sequential_pipeline(
        ...     sub_graphs=[
        ...         time_agent, weather_agent, user_agent
        ...     ],
        ...     state_schema=AgentState,
        ...     graph_name="sequential_agents_pipeline",
        ... )
        >>>
        >>> response = graph.invoke({"messages": [HumanMessage("Hello")]})
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

    graph.add_sequence(
        [
            (compiled_subgraphs[i].name, compiled_subgraphs[i])
            for i in range(len(compiled_subgraphs))
        ]
    )
    graph.add_edge("__start__", compiled_subgraphs[0].name)
    return graph.compile(
        name=graph_name or "sequential graph",
        checkpointer=checkpointer,
        store=store,
        cache=cache,
    )
