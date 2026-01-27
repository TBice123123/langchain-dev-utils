from typing import Optional

from langgraph.cache.base import BaseCache
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer
from langgraph.typing import ContextT, InputT, OutputT, StateT

from langchain_dev_utils._utils import _transform_node_to_tuple

from .types import Node


def create_sequential_graph(
    nodes: list[Node],
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
    Create a sequential graph from a list of nodes.

    This function lets you build a sequential StateGraph simply by writing the corresponding Nodes.

    Args:
        nodes: List of nodes to execute sequentially
        state_schema: state schema of the final state graph
        graph_name: Name of the final state graph
        context_schema: context schema of the final state graph
        input_schema: input schema of the final state graph
        output_schema: output schema of the final state graph
        checkpointer: Optional LangGraph checkpointer for the final state graph
        store: Optional LangGraph store for the final state graph
        cache: Optional LangGraph cache for the final state graph
    Returns:
        CompiledStateGraph[StateT, ContextT, InputT, OutputT]: Compiled state graph.

    Example:
        # Basic sequential graph with multiple specialized agents:
        >>> from langchain_dev_utils.graph.sequential import create_sequential_graph
        >>>
        >>> graph = create_sequential_graph(
        ...     nodes=[
        ...         node1, node2, node3
        ...     ],
        ...     state_schema=State,
        ...     graph_name="sequential_graph",
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

    node_list = []
    for node in nodes:
        node = _transform_node_to_tuple(node)
        node_list.append(node)

    graph.add_sequence(node_list)
    first_node_name, _ = node_list[0]
    graph.add_edge("__start__", first_node_name)
    return graph.compile(
        name=graph_name or "sequential graph",
        checkpointer=checkpointer,
        store=store,
        cache=cache,
    )
