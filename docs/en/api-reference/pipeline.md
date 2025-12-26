# Pipeline Module API Reference Documentation

## create_sequential_pipeline

Combines multiple subgraphs with the same state in a sequential manner.

### Function Signature

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

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| sub_graphs | list[SubGraph] | Yes | - | List of state graphs to combine |
| state_schema | type[StateT] | Yes | - | State Schema of the final generated graph |
| graph_name | Optional[str] | No | None | Name of the final generated graph |
| context_schema | type[ContextT] \| None | No | None | Context Schema of the final generated graph |
| input_schema | type[InputT] \| None | No | None | Input Schema of the final generated graph |
| output_schema | type[OutputT] \| None | No | None | Output Schema of the final generated graph |
| checkpointer | Checkpointer \| None | No | None | Checkpointer of the final generated graph |
| store | BaseStore \| None | No | None | Store of the final generated graph |
| cache | BaseCache \| None | No | None | Cache of the final generated graph |

### Example

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

---

## create_parallel_pipeline

Combines multiple subgraphs with the same state in a parallel manner.

### Function Signature

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

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| sub_graphs | list[SubGraph] | Yes | - | List of state graphs to combine |
| state_schema | type[StateT] | Yes | - | State Schema of the final generated graph |
| graph_name | Optional[str] | No | None | Name of the final generated graph |
| branches_fn | Optional[Union[Callable[..., list[Send]], Callable[..., Awaitable[list[Send]]]]] | No | None | Parallel branch function, returns a list of `Send` objects to control parallel execution |
| context_schema | type[ContextT] \| None | No | None | Context Schema of the final generated graph |
| input_schema | type[InputT] \| None | No | None | Input Schema of the final generated graph |
| output_schema | type[OutputT] \| None | No | None | Output Schema of the final generated graph |
| checkpointer | Checkpointer \| None | No | None | Checkpointer of the final generated graph |
| store | BaseStore \| None | No | None | Store of the final generated graph |
| cache | BaseCache \| None | No | None | Cache of the final generated graph |

### Example

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