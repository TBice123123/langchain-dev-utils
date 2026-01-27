# Graph 模块 API 参考文档

## create_sequential_graph

将多个节点以串行方式组合成一个状态图。

### 函数签名

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

### 参数

| 参数 | 类型 | 必填 | 默认值 | 描述 |
|------|------|------|--------|------|
| nodes | list[Node] | 是 | - | 要组合的节点列表，可为节点函数或由节点名称与节点函数组成的二元组。 |
| state_schema | type[StateT] | 是 | - | 最终生成图的 State Schema |
| graph_name | Optional[str] | 否 | None | 最终生成图的名称 |
| context_schema | type[ContextT] \| None | 否 | None | 最终生成图的 Context Schema |
| input_schema | type[InputT] \| None | 否 | None | 最终生成图的输入 Schema |
| output_schema | type[OutputT] \| None | 否 | None | 最终生成图的输出 Schema |
| checkpointer | Checkpointer \| None | 否 | None | 最终生成图的 Checkpointer |
| store | BaseStore \| None | 否 | None | 最终生成图的 Store |
| cache | BaseCache \| None | 否 | None | 最终生成图的 Cache |

### 示例

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

将多个节点以并行方式组合成一个状态图。

### 函数签名

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

### 参数

| 参数 | 类型 | 必填 | 默认值 | 描述 |
|------|------|------|--------|------|
| nodes | list[Node] | 是 | - | 要组合的节点列表，可为节点函数或由节点名称与节点函数组成的二元组。 |
| state_schema | type[StateT] | 是 | - | 最终生成图的 State Schema |
| graph_name | Optional[str] | 否 | None | 最终生成图的名称 |
| branches_fn | Optional[Union[Callable[..., list[Send]], Callable[..., Awaitable[list[Send]]]]] | 否 | None | 并行分支函数，返回 Send 列表控制并行执行 |
| context_schema | type[ContextT] \| None | 否 | None | 最终生成图的 Context Schema |
| input_schema | type[InputT] \| None | 否 | None | 最终生成图的输入 Schema |
| output_schema | type[OutputT] \| None | 否 | None | 最终生成图的输出 Schema |
| checkpointer | Checkpointer \| None | 否 | None | 最终生成图的 Checkpointer |
| store | BaseStore \| None | 否 | None | 最终生成图的 Store |
| cache | BaseCache \| None | 否 | None | 最终生成图的 Cache |


### 示例

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

## Node 类型

```python
Node = StateNode | tuple[str, StateNode]
```