# Pipeline Module API Reference

## create_sequential_pipeline

Combines multiple subgraphs with identical states in a sequential manner.

```python
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
```

**Parameters:**

- `sub_graphs`: List of SubGraph type, required. List of state graphs to be combined.
- `state_schema`: StateT type, required. State Schema for the final generated graph.
- `graph_name`: Optional string. Name of the final generated graph.
- `context_schema`: ContextT type or None, optional. Context Schema for the final generated graph.
- `input_schema`: InputT type or None, optional. Input Schema for the final generated graph.
- `output_schema`: OutputT type or None, optional. Output Schema for the final generated graph.
- `checkpointer`: Optional LangGraph checkpointer for the final constructed graph
- `store`: Optional LangGraph store for the final constructed graph
- `cache`: Optional LangGraph cache for the final constructed graph

**Return Value:** CompiledStateGraph type, the created sequential state graph.

**Example:**

```python
create_sequential_pipeline(
    sub_graphs=[graph1, graph2],
    state_schema=State,
    graph_name="sequential_pipeline",
    context_schema=Context,
    input_schema=Input,
    output_schema=Output,
)
```

## create_parallel_pipeline

Combines multiple subgraphs with identical states in a parallel manner.

```python
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
```

**Parameters:**

- `sub_graphs`: List of SubGraph type, required. List of state graphs to be combined.
- `state_schema`: StateT type, required. State Schema for the final generated graph.
- `graph_name`: Optional string. Name of the final generated graph.
- `branches_fn`: Optional callable type. Parallel branch function that returns a list of Send to control parallel execution.
- `context_schema`: ContextT type or None, optional. Context Schema for the final generated graph.
- `input_schema`: InputT type or None, optional. Input Schema for the final generated graph.
- `output_schema`: OutputT type or None, optional. Output Schema for the final generated graph.
- `checkpointer`: Optional LangGraph checkpointer for the final constructed graph
- `store`: Optional LangGraph store for the final constructed graph
- `cache`: Optional LangGraph cache for the final constructed graph.

**Return Value:** CompiledStateGraph type, the created parallel state graph.

**Example:**

```python
create_parallel_pipeline(
    sub_graphs=[graph1, graph2],
    state_schema=State,
    graph_name="parallel_pipeline",
    branches_fn=lambda state: [Send("graph1", state), Send("graph2", state)],
    context_schema=Context,
    input_schema=Input,
    output_schema=Output,
)
```
