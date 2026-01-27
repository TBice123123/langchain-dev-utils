# Graph Module API Reference

## create_sequential_graph

Combines multiple nodes into a state graph in a serial (sequential) manner.

### Function Signature

```python
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
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| nodes | list[Node] | Yes | - | The list of nodes to combine, which can be node functions or tuples consisting of a node name and a node function. |
| state_schema | type[StateT] | Yes | - | The State Schema of the final generated graph. |
| graph_name | Optional[str] | No | None | The name of the final generated graph. |
| context_schema | type[ContextT] \| None | No | None | The Context Schema of the final generated graph. |
| input_schema | type[InputT] \| None | No | None | The Input Schema of the final generated graph. |
| output_schema | type[OutputT] \| None | No | None | The Output Schema of the final generated graph. |
| checkpointer | Checkpointer \| None | No | None | The Checkpointer of the final generated graph. |
| store | BaseStore \| None | No | None | The Store of the final generated graph. |
| cache | BaseCache \| None | No | None | The Cache of the final generated graph. |

### Example

```python
create_sequential_graph(
    nodes=[node1, node2],
    state_schema=State,
    graph_name="sequential_pipeline",
    context_schema=Context,
    input_schema=Input,
    output_schema=Output,
)
```

---

## create_parallel_graph

Combines multiple nodes into a state graph in a parallel manner.

### Function Signature

```python
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
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| nodes | list[Node] | Yes | - | The list of nodes to combine, which can be node functions or tuples consisting of a node name and a node function. |
| state_schema | type[StateT] | Yes | - | The State Schema of the final generated graph. |
| graph_name | Optional[str] | No | None | The name of the final generated graph. |
| branches_fn | Optional[Union[Callable[..., list[Send]], Callable[..., Awaitable[list[Send]]]]] | No | None | The parallel branch function that returns a list of Send objects to control parallel execution. |
| context_schema | type[ContextT] \| None | No | None | The Context Schema of the final generated graph. |
| input_schema | type[InputT] \| None | No | None | The Input Schema of the final generated graph. |
| output_schema | type[OutputT] \| None | No | None | The Output Schema of the final generated graph. |
| checkpointer | Checkpointer \| None | No | None | The Checkpointer of the final generated graph. |
| store | BaseStore \| None | No | None | The Store of the final generated graph. |
| cache | BaseCache \| None | No | None | The Cache of the final generated graph. |

### Example

```python
create_parallel_graph(
    nodes=[node1, node2],
    state_schema=State,
    graph_name="parallel_pipeline",
    branches_fn=lambda state: [Send("node1", state), Send("node2", state)],
    context_schema=Context,
    input_schema=Input,
    output_schema=Output,
)
```

## Node Type

```python
Node = StateNode | tuple[str, StateNode]
```