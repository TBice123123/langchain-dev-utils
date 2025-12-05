# Agent 模块的 API 参考

## create_agent

预构建智能体函数，提供与 langchain 官方 `create_agent` 完全相同的功能，但拓展了字符串指定模型。

```python
def create_agent(  # noqa: PLR0915
    model: str,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware[AgentState[ResponseT], ContextT]] = (),
    response_format: ResponseFormat[ResponseT] | type[ResponseT] | None = None,
    state_schema: type[AgentState[ResponseT]] | None = None,
    context_schema: type[ContextT] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,
) -> CompiledStateGraph:
```

**参数说明：**

- `model`：字符串类型，必填，可由 `load_chat_model` 加载的模型标识符字符串。可指定为 "provider:model-name" 格式
- `tools`：BaseTool、Callable 或字典序列，或 ToolNode 类型，必填，智能体可用的工具列表
- `system_prompt`：可选字符串类型，智能体的自定义系统提示词
- `middleware`：可选 AgentMiddleware 类型，智能体的中间件
- `response_format`：可选 ResponseFormat 类型，智能体的响应格式
- `state_schema`：可选 StateSchemaType 类型，智能体的状态模式
- `context_schema`：可选任意类型，智能体的上下文模式
- `checkpointer`：可选 Checkpointer 类型，状态持久化的检查点
- `store`：可选 BaseStore 类型，数据持久化的存储
- `interrupt_before`：可选字符串列表类型，执行前要中断的节点
- `interrupt_after`：可选字符串列表类型，执行后要中断的节点
- `debug`：布尔类型，可选，启用调试模式，默认为 False
- `name`：可选字符串类型，智能体名称
- `cache`：可选 BaseCache 类型，缓存

**返回值：** CompiledStateGraph 类型

**注意：** 此函数提供与 `langchain` 官方 `create_agent` 完全相同的功能，但拓展了模型选择。主要区别在于 `model` 参数必须是可由 `load_chat_model` 函数加载的字符串，允许使用注册的模型提供者进行更灵活的模型选择。

**示例：**

```python
agent = create_agent(model="vllm:qwen3-4b", tools=[get_current_time])
```

## wrap_agent_as_tool

```python
def wrap_agent_as_tool(
    agent: CompiledStateGraph,
    tool_name: Optional[str] = None,
    tool_description: Optional[str] = None,
    pre_input_hooks: Optional[
        tuple[
            Callable[[str, ToolRuntime], str],
            Callable[[str, ToolRuntime], Awaitable[str]],
        ]
        | Callable[[str, ToolRuntime], str]
    ] = None,
    post_output_hooks: Optional[
        tuple[
            Callable[[str, list[AnyMessage], ToolRuntime], Any],
            Callable[[str, list[AnyMessage], ToolRuntime], Awaitable[Any]],
        ]
        | Callable[[str, list[AnyMessage], ToolRuntime], Any]
    ] = None,
) -> BaseTool:
```

**参数说明：**

- `agent`：CompiledStateGraph 类型，必填，智能体
- `tool_name`：可选字符串类型，工具名称
- `tool_description`：可选字符串类型，工具描述
- `pre_input_hooks`：可选元组类型或者函数类型，Agent 输入预处理函数
- `post_output_hooks`：可选元组类型或者函数类型，Agent 输出后处理函数

**返回值：** BaseTool 类型，工具实例

**示例：**

```python
tool = wrap_agent_as_tool(agent)
```

## create_write_plan_tool

创建用于写计划或者更新计划的工具。

```python
def create_write_plan_tool(
    description: Optional[str] = None,
    message_key: Optional[str] = None,
) -> BaseTool:
```

**参数说明：**

- `description`：可选字符串类型，工具描述
- `message_key`：可选字符串类型，用于更新 messages 的键，默认 "messages"

**返回值：** BaseTool 类型，工具实例

**示例：**

```python
write_plan_tool = create_write_plan_tool()
```

## create_finish_sub_plan_tool

创建用于完成子任务后更新执行状态的工具。

```python
def create_finish_sub_plan_tool(
    description: Optional[str] = None,
    message_key: Optional[str] = None,
) -> BaseTool:
```

**参数说明：**

- `description`：可选字符串类型，工具描述
- `message_key`：可选字符串类型，用于更新 messages 的键，默认 "messages"

**返回值：** BaseTool 类型，工具实例

**示例：**

```python
finish_sub_plan_tool = create_finish_sub_plan_tool()
```

## create_read_plan_tool

创建用于读取执行状态的工具。

```python
def create_read_plan_tool(
    description: Optional[str] = None,
) -> BaseTool:
```

**参数说明：**

- `description`：可选字符串类型，工具描述

**返回值：** BaseTool 类型，工具实例

**示例：**

```python
read_plan_tool = create_read_plan_tool()
```

## SummarizationMiddleware

用于智能体上下文摘要的中间件。

```python
class SummarizationMiddleware(_SummarizationMiddleware):
    def __init__(
        self,
        model: str,
        *,
        trigger: ContextSize | list[ContextSize] | None = None,
        keep: ContextSize = ("messages", _DEFAULT_MESSAGES_TO_KEEP),
        token_counter: TokenCounter = count_tokens_approximately,
        summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
        trim_tokens_to_summarize: int | None = _DEFAULT_TRIM_TOKEN_LIMIT,
        **deprecated_kwargs: Any,
    ) -> None:
```

**参数说明：**

- `model`：字符串类型，必填，可由 `load_chat_model` 加载的模型标识符字符串。可指定为 "provider:model-name" 格式
- `trigger`：可选 ContextSize 类型或者列表类型，触发摘要的上下文大小
- `keep`：可选 ContextSize 类型，保留的上下文大小
- `token_counter`：可选 TokenCounter 类型，token 计数器
- `summary_prompt`：可选字符串类型，摘要提示词
- `trim_tokens_to_summarize`：可选整数类型，摘要前要截取的 token 数

**示例：**

```python
summarization_middleware = SummarizationMiddleware(model="vllm:qwen3-4b")
```

## LLMToolSelectorMiddleware

用于智能体工具选择的中间件。

```python
class LLMToolSelectorMiddleware(_LLMToolSelectorMiddleware):
    def __init__(
        self,
        *,
        model: str,
        system_prompt: Optional[str] = None,
        max_tools: Optional[int] = None,
        always_include: Optional[list[str]] = None,
    ) -> None:
```

**参数说明：**

- `model`：字符串类型，必填，可由 `load_chat_model` 加载的模型标识符字符串。可指定为 "provider:model-name" 格式
- `system_prompt`：可选字符串类型，系统提示词
- `max_tools`：可选整数类型，最大工具数
- `always_include`：可选字符串列表类型，总是包含的工具

**示例：**

```python
llm_tool_selector_middleware = LLMToolSelectorMiddleware(model="vllm:qwen3-4b")
```

## PlanMiddleware

用于智能体计划管理的中间件。

```python
class PlanMiddleware(AgentMiddleware):
    state_schema = PlanState
    def __init__(
        self,
        *,
        system_prompt: Optional[str] = None,
        write_plan_tool_description: Optional[str] = None,
        finish_sub_plan_tool_description: Optional[str] = None,
        read_plan_tool_description: Optional[str] = None,
        use_read_plan_tool: bool = True
    ) -> None:
```

**参数说明：**

- `system_prompt`：可选字符串类型，系统提示词
- `write_plan_tool_description`：可选字符串类型，写计划工具的描述
- `finish_sub_plan_tool_description`：可选字符串类型，完成子计划工具的描述
- `read_plan_tool_description`：可选字符串类型，读计划工具的描述
- `use_read_plan_tool`：可选布尔类型，是否使用读计划工具

**示例：**

```python
plan_middleware = PlanMiddleware()
```

## ModelFallbackMiddleware

用于智能体模型回退的中间件。

```python
class ModelFallbackMiddleware(_ModelFallbackMiddleware):
    def __init__(
        self,
        first_model: str,
        *additional_models: str,
    ) -> None:
```

**参数说明：**

- `first_model`：字符串类型，必填，可由 `load_chat_model` 加载的模型标识符字符串。可指定为 "provider:model-name" 格式
- `additional_models`：可选字符串列表类型，备用模型列表

**示例：**

```python
model_fallback_middleware = ModelFallbackMiddleware(
    "vllm:qwen3-4b",
    "vllm:qwen3-8b"
)
```

## LLMToolEmulator

用于使用大模型来模拟工具调用的中间件。

```python
class LLMToolEmulator(_LLMToolEmulator):
    def __init__(
        self,
        *,
        model: str,
        tools: list[str | BaseTool] | None = None,
    ) -> None:
```

**参数说明：**

- `model`：字符串类型，必填，可由 `load_chat_model` 加载的模型标识符字符串。可指定为 "provider:model-name" 格式
- `tools`：可选 BaseTool 列表类型，工具列表

**示例：**

```python
llm_tool_emulator = LLMToolEmulator(model="vllm:qwen3-4b", tools=[get_current_time])
```

## ModelRouterMiddleware

用于根据输入内容动态路由到合适模型的中间件。

```python
class ModelRouterMiddleware(AgentMiddleware):
    state_schema = ModelRouterState
    def __init__(
        self,
        router_model: str | BaseChatModel,
        model_list: list[ModelDict],
        router_prompt: Optional[str] = None,
    ) -> None:
```

**参数说明：**

- `router_model`: 用于路由的模型,接收字符串类型（使用`load_chat_model`加载）或者直接传入 ChatModel
- `model_list`：模型列表，每个模型需要包含 `model_name` 和 `model_description` 两个键，同时也可以选择性地包含 `tools`、`model_kwargs` 、`model_system_prompt` 这三个键，分别代表模型可用的工具（如果不传则默认是使用全部工具）、传递给模型的额外参数（例如：temperature、top_p 等）和该模型的系统提示词。
- `router_prompt`: 路由模型的提示词，如果为 None 则使用默认的提示词

**示例：**

```python
model_router_middleware = ModelRouterMiddleware(
    router_model="vllm:qwen3-4b",
    model_list=[
        {
            "model_name": "vllm:qwen3-4b",
            "model_description": "适合普通任务，如对话、文本生成等"
        },
        {
            "model_name": "vllm:qwen3-8b",
            "model_description": "适合复杂任务，如代码生成、数据分析等",
        },
    ]
)
```

## ToolCallRepairMiddleware

用于修复无效工具调用的中间件。

```python
class ToolCallRepairMiddleware(AgentMiddleware):
```

**示例：**

```python
tool_call_repair_middleware = ToolCallRepairMiddleware()
```

## PlanState

用于 Plan 的状态 Schema。

```python
class Plan(TypedDict):
    content: str
    status: Literal["pending", "in_progress", "done"]


class PlanState(AgentState):
    plan: NotRequired[list[Plan]]
```

- `plan`：可选列表类型，计划列表
- `plan.content`：计划内容
- `plan.status`：计划状态,取值为`pending`、`in_progress`、`done`

## ModelDict

模型列表的类型。

```python
class ModelDict(TypedDict):
    model_name: str
    model_description: str
    tools: NotRequired[list[BaseTool | dict[str, Any]]]
    model_kwargs: NotRequired[dict[str, Any]]
    model_system_prompt: NotRequired[str]
```

## SelectModel

用于选择模型的工具类。

```python
class SelectModel(BaseModel):
    """Tool for model selection - Must call this tool to return the finally selected model"""

    model_name: str = Field(
        ...,
        description="Selected model name (must be the full model name, for example, openai:gpt-4o)",
    )
```
