# Pipeline 模块的 API 参考

## create_sequential_pipeline

将多个状态相同的子图以串行方式组合。

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

**参数说明：**

- `sub_graphs`：SubGraph 列表类型，必填，要组合的状态图列表
- `state_schema`：StateT 类型，必填，最终生成图的 State Schema
- `graph_name`：可选字符串类型，最终生成图的名称
- `context_schema`：ContextT 类型或 None，可选，最终生成图的 Context Schema
- `input_schema`：InputT 类型或 None，可选，最终生成图的输入 Schema
- `output_schema`：OutputT 类型或 None，可选，最终生成图的输出 Schema
- `checkpointer`：LangGraph checkpointer 类型或 None，最终生成图的 Checkpointer
- `store`：LangGraph store 类型或 None，最终生成图的 Store
- `cache`：LangGraph cache 类型或 None，最终生成图的 Cache

**返回值：** CompiledStateGraph 类型，创建的串行状态图

**示例：**

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

将多个状态相同的子图以并行方式组合。

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

**参数说明：**

- `sub_graphs`：SubGraph 列表类型，必填，要组合的状态图列表
- `state_schema`：StateT 类型，必填，最终生成图的 State Schema
- `graph_name`：可选字符串类型，最终生成图的名称
- `branches_fn`：可选可调用类型，并行分支函数，返回 Send 列表控制并行执行
- `context_schema`：ContextT 类型或 None，可选，最终生成图的 Context Schema
- `input_schema`：InputT 类型或 None，可选，最终生成图的输入 Schema
- `output_schema`：OutputT 类型或 None，可选，最终生成图的输出 Schema
- `checkpointer`：LangGraph checkpointer 类型或 None，最终生成图的 Checkpointer
- `store`：LangGraph store 类型或 None，最终生成图的 Store
- `cache`：LangGraph cache 类型或 None，最终生成图的 Cache

**返回值：** CompiledStateGraph 类型，创建的并行状态图

**示例：**

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
