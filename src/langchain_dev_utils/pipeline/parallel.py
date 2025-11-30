from typing import Awaitable, Callable, Optional, Union

from langgraph.cache.base import BaseCache
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer, Send
from langgraph.typing import ContextT, InputT, OutputT, StateT

from .types import SubGraph


def create_parallel_pipeline(
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
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    cache: BaseCache | None = None,
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
        branches_fn: Optional function to determine which sub-graphs to execute
            in parallel
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
        # Basic parallel pipeline: multiple specialized agents run concurrently
        >>> from langchain_dev_utils.pipeline import create_parallel_pipeline
        >>>
        >>> graph = create_parallel_pipeline(
        ...     sub_graphs=[
        ...        time_agent, weather_agent, user_agent
        ...     ],
        ...     state_schema=AgentState,
        ...     graph_name="parallel_agents_pipeline",
        ... )
        >>>
        >>> response = graph.invoke({"messages": [HumanMessage("Hello")]})

        # Dynamic parallel pipeline: decide which agents to run based on conditional branches
        >>> graph = create_parallel_pipeline(
        ...     sub_graphs=[
        ...         time_agent, weather_agent, user_agent
        ...     ],
        ...     state_schema=AgentState,
        ...     branches_fn=lambda state: [
        ...         Send("weather_agent", arg={"messages": [HumanMessage("Get current weather in New York")]}),
        ...         Send("time_agent", arg={"messages": [HumanMessage("Get current time")]}),
        ...     ],
        ...     graph_name="dynamic_parallel_pipeline",
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

    for sub_graph in compiled_subgraphs:
        graph.add_node(sub_graph.name, sub_graph)

    if branches_fn:
        graph.add_conditional_edges(
            "__start__",
            branches_fn,
            [subgraph.name for subgraph in compiled_subgraphs],
        )
        return graph.compile(
            name=graph_name or "parallel graph",
            checkpointer=checkpointer,
            store=store,
            cache=cache,
        )
    else:
        for i in range(len(compiled_subgraphs)):
            graph.add_edge("__start__", compiled_subgraphs[i].name)
        return graph.compile(
            name=graph_name or "parallel graph",
            checkpointer=checkpointer,
            store=store,
            cache=cache,
        )
