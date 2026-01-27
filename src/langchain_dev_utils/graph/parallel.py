from typing import Awaitable, Callable, Optional, Union, cast

from langgraph.cache.base import BaseCache
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph, StateNode
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer, Send
from langgraph.typing import ContextT, InputT, OutputT, StateT

from langchain_dev_utils._utils import _transform_node_to_tuple

from .types import Node


def create_parallel_graph(
    nodes: list[Node],
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
    Create a parallel graph from a list of nodes.

    This function lets you build a parallel StateGraph simply by writing the corresponding Nodes.

    Args:
        nodes: List of nodes to execute in parallel
        state_schema: state schema of the final state graph
        graph_name: Name of the final state graph
        branches_fn: Optional function to determine which nodes to execute
            in parallel
        context_schema: context schema of the final state graph
        input_schema: input schema of the final state graph
        output_schema: output schema of the final state graph
        checkpointer: Optional LangGraph checkpointer for the final state graph
        store: Optional LangGraph store for the final state graph
        cache: Optional LangGraph cache for the final state graph

    Returns:
        CompiledStateGraph[StateT, ContextT, InputT, OutputT]: Compiled state graph

    Example:
        # Basic parallel pipeline: multiple specialized agents run concurrently
        >>> from langchain_dev_utils.graph import create_parallel_graph
        >>>
        >>> graph = create_parallel_graph(
        ...     nodes=[
        ...        node1, node2, node3
        ...     ],
        ...     state_schema=StateT,
        ...     graph_name="parallel_graph",
        ... )
        >>>
        >>> response = graph.invoke({"messages": [HumanMessage("Hello")]})

        # Dynamic parallel pipeline: decide which nodes to run based on conditional branches
        >>> graph = create_parallel_graph(
        ...     nodes=[
        ...         node1, node2, node3
        ...     ],
        ...     state_schema=StateT,
        ...     branches_fn=lambda state: [
        ...         Send("node1", arg={"messages": [HumanMessage("Hello")]}),
        ...         Send("node2", arg={"messages": [HumanMessage("Hello")]}),
        ...     ],
        ...     graph_name="parallel_graph",
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

    node_list: list[tuple[str, StateNode]] = []

    for node in nodes:
        node_list.append(_transform_node_to_tuple(node))

    if branches_fn:
        for name, node in node_list:
            node = cast(StateNode[StateT, ContextT], node)
            graph.add_node(name, node)
        graph.add_conditional_edges(
            "__start__",
            branches_fn,
            [node_name for node_name, _ in node_list],
        )
        return graph.compile(
            name=graph_name or "parallel graph",
            checkpointer=checkpointer,
            store=store,
            cache=cache,
        )
    else:
        for node_name, node in node_list:
            node = cast(StateNode[StateT, ContextT], node)
            graph.add_node(node_name, node)
            graph.add_edge("__start__", node_name)
        return graph.compile(
            name=graph_name or "parallel graph",
            checkpointer=checkpointer,
            store=store,
            cache=cache,
        )
