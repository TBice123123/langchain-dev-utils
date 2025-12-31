# Agent 模块 API 参考文档

## create_agent

创建一个智能体，提供与 langchain 官方 `create_agent` 完全相同的功能，但拓展了字符串指定模型。

### 函数签名

```python
def create_agent(  # noqa: PLR0915
    model: str,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    response_format: ResponseFormat[ResponseT] | type[ResponseT] | None = None,
    middleware: Sequence[AgentMiddleware[StateT_co, ContextT]] = (),
    state_schema: type[AgentState[ResponseT]] | None = None,
    context_schema: type[ContextT] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,
) -> CompiledStateGraph[
    AgentState[ResponseT], ContextT, _InputAgentState, _OutputAgentState[ResponseT]
]:
```

### 参数

| 参数 | 类型 | 必填 | 默认值 | 描述 |
|------|------|------|--------|------|
| model | str | 是 | - | 可由 `load_chat_model` 加载的模型标识符字符串。可指定为 "provider:model-name" 格式 |
| tools | Sequence[BaseTool \| Callable \| dict[str, Any]] \| None | 否 | None | 智能体可用的工具列表 |
| system_prompt | str \| SystemMessage \| None | 否 | None | 智能体的自定义系统提示词 |
| middleware | Sequence[AgentMiddleware[AgentState[ResponseT], ContextT]] | 否 | () | 智能体的中间件 |
| response_format | ResponseFormat[ResponseT] \| type[ResponseT] \| None | 否 | None | 智能体的响应格式 |
| state_schema | type[AgentState[ResponseT]] \| None | 否 | None | 智能体的状态模式 |
| context_schema | type[ContextT] \| None | 否 | None | 智能体的上下文模式 |
| checkpointer | Checkpointer \| None | 否 | None | 状态持久化的检查点 |
| store | BaseStore \| None | 否 | None | 数据持久化的存储 |
| interrupt_before | list[str] \| None | 否 | None | 执行前要中断的节点 |
| interrupt_after | list[str] \| None | 否 | None | 执行后要中断的节点 |
| debug | bool | 否 | False | 启用调试模式 |
| name | str \| None | 否 | None | 智能体名称 |
| cache | BaseCache \| None | 否 | None | 缓存 |


### 注意事项

此函数提供与 `langchain` 官方 `create_agent` 完全相同的功能，但拓展了模型选择。主要区别在于 `model` 参数必须是可由 `load_chat_model` 函数加载的字符串，允许使用注册的模型提供者进行更灵活的模型选择。

### 示例

```python
agent = create_agent(model="vllm:qwen3-4b", tools=[get_current_time])
```

---

## wrap_agent_as_tool

将智能体包装为工具。

### 函数签名

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
) -> BaseTool
```

### 参数

| 参数 | 类型 | 必填 | 默认值 | 描述 |
|------|------|------|--------|------|
| agent | CompiledStateGraph | 是 | - | 智能体 |
| tool_name | Optional[str] | 否 | None | 工具名称 |
| tool_description | Optional[str] | 否 | None | 工具描述 |
| pre_input_hooks | Optional[tuple[Callable[[str, ToolRuntime], str], Callable[[str, ToolRuntime], Awaitable[str]]] \| Callable[[str, ToolRuntime], str]] | 否 | None | Agent 输入预处理函数 |
| post_output_hooks | Optional[tuple[Callable[[str, list[AnyMessage], ToolRuntime], Any], Callable[[str, list[AnyMessage], ToolRuntime], Awaitable[Any]]] \| Callable[[str, list[AnyMessage], ToolRuntime], Any]] | 否 | None | Agent 输出后处理函数 |


### 示例

```python
tool = wrap_agent_as_tool(agent)
```

---

## wrap_all_agents_as_tool

将所有智能体包装为单个工具。

### 函数签名

```python
def wrap_all_agents_as_tool(
    agents: list[CompiledStateGraph],
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


### 参数

| 参数 | 类型 | 必填 | 默认值 | 描述 |
|------|------|------|--------|------|
| agents | list[CompiledStateGraph] | 是 | - | 智能体列表(至少包含2个，且每个智能体必须有唯一的名称) |
| tool_name | Optional[str] | 否 | None | 工具名称 |
| tool_description | Optional[str] | 否 | None | 工具描述 |
| pre_input_hooks | Optional[tuple[Callable[[str, ToolRuntime], str], Callable[[str, ToolRuntime], Awaitable[str]]] \| Callable[[str, ToolRuntime], str]] | 否 | None | Agent 输入预处理函数 |
| post_output_hooks | Optional[tuple[Callable[[str, list[AnyMessage], ToolRuntime], Any], Callable[[str, list[AnyMessage], ToolRuntime], Awaitable[Any]]] \| Callable[[str, list[AnyMessage], ToolRuntime], Any]] | 否 | None | Agent 输出后处理函数 |

### 示例

```python
tool = wrap_all_agents_as_tool([time_agent, weather_agent])
```

---

## SummarizationMiddleware

用于智能体上下文摘要的中间件。

### 类定义

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
    ) -> None
```

### 参数

| 参数 | 类型 | 必填 | 默认值 | 描述 |
|------|------|------|--------|------|
| model | str | 是 | - | 可由 `load_chat_model` 加载的模型标识符字符串。可指定为 "provider:model-name" 格式 |
| trigger | ContextSize \| list[ContextSize] \| None | 否 | None | 触发摘要的上下文大小 |
| keep | ContextSize | 否 | ("messages", _DEFAULT_MESSAGES_TO_KEEP) | 保留的上下文大小 |
| token_counter | TokenCounter | 否 | count_tokens_approximately | token 计数器 |
| summary_prompt | str | 否 | DEFAULT_SUMMARY_PROMPT | 摘要提示词 |
| trim_tokens_to_summarize | int \| None | 否 | _DEFAULT_TRIM_TOKEN_LIMIT | 摘要前要截取的 token 数 |

### 示例

```python
summarization_middleware = SummarizationMiddleware(model="vllm:qwen3-4b")
```

---

## LLMToolSelectorMiddleware

用于智能体工具选择的中间件。

### 类定义

```python
class LLMToolSelectorMiddleware(_LLMToolSelectorMiddleware):
    def __init__(
        self,
        *,
        model: str,
        system_prompt: Optional[str] = None,
        max_tools: Optional[int] = None,
        always_include: Optional[list[str]] = None,
    ) -> None
```

### 参数

| 参数 | 类型 | 必填 | 默认值 | 描述 |
|------|------|------|--------|------|
| model | str | 是 | - | 可由 `load_chat_model` 加载的模型标识符字符串。可指定为 "provider:model-name" 格式 |
| system_prompt | Optional[str] | 否 | None | 系统提示词 |
| max_tools | Optional[int] | 否 | None | 最大工具数 |
| always_include | Optional[list[str]] | 否 | None | 总是包含的工具 |

### 示例

```python
llm_tool_selector_middleware = LLMToolSelectorMiddleware(model="vllm:qwen3-4b")
```

---

## PlanMiddleware

用于智能体计划管理的中间件。

### 类定义

```python
class PlanMiddleware(AgentMiddleware):
    state_schema = PlanState
    def __init__(
        self,
        *,
        system_prompt: Optional[str] = None,
        custom_plan_tool_descriptions: Optional[PlanToolDescription] = None,
        use_read_plan_tool: bool = True,
    ) -> None:
```

### 参数

| 参数 | 类型 | 必填 | 默认值 | 描述 |
|------|------|------|--------|------|
| system_prompt | Optional[str] | 否 | None | 系统提示词 |
| custom_plan_tool_descriptions | Optional[PlanToolDescription] | 否 | None | 自定义计划工具的描述 |
| use_read_plan_tool | bool | 否 | True | 是否使用读计划工具 |


### 示例

```python
plan_middleware = PlanMiddleware()
```

---

## ModelFallbackMiddleware

用于智能体模型回退的中间件。

### 类定义

```python
class ModelFallbackMiddleware(_ModelFallbackMiddleware):
    def __init__(
        self,
        first_model: str,
        *additional_models: str,
    ) -> None
```

### 参数

| 参数 | 类型 | 必填 | 默认值 | 描述 |
|------|------|------|--------|------|
| first_model | str | 是 | - | 可由 `load_chat_model` 加载的模型标识符字符串。可指定为 "provider:model-name" 格式 |
| additional_models | str | 否 | - | 备用模型列表 |

### 示例

```python
model_fallback_middleware = ModelFallbackMiddleware(
    "vllm:qwen3-4b",
    "vllm:qwen3-8b"
)
```

---

## LLMToolEmulator

用于使用大模型来模拟工具调用的中间件。

### 类定义

```python
class LLMToolEmulator(_LLMToolEmulator):
    def __init__(
        self,
        *,
        model: str,
        tools: list[str | BaseTool] | None = None,
    ) -> None
```

### 参数

| 参数 | 类型 | 必填 | 默认值 | 描述 |
|------|------|------|--------|------|
| model | str | 是 | - | 可由 `load_chat_model` 加载的模型标识符字符串。可指定为 "provider:model-name" 格式 |
| tools | list[str \| BaseTool] \| None | 否 | None | 工具列表 |

### 示例

```python
llm_tool_emulator = LLMToolEmulator(model="vllm:qwen3-4b", tools=[get_current_time])
```

---

## ModelRouterMiddleware

用于根据输入内容动态路由到合适模型的中间件。

### 类定义

```python
class ModelRouterMiddleware(AgentMiddleware):
    state_schema = ModelRouterState
    def __init__(
        self,
        router_model: str | BaseChatModel,
        model_list: list[ModelDict],
        router_prompt: Optional[str] = None,
    ) -> None
```

### 参数

| 参数 | 类型 | 必填 | 默认值 | 描述 |
|------|------|------|--------|------|
| router_model | str \| BaseChatModel | 是 | - | 用于路由的模型，接收字符串类型（使用`load_chat_model`加载）或者直接传入 ChatModel |
| model_list | list[ModelDict] | 是 | - | 模型列表，每个模型需要包含 `model_name` 和 `model_description` 两个键，同时也可以选择性地包含 `tools`、`model_kwargs`、`model_instance`、`model_system_prompt` 这四个键 |
| router_prompt | Optional[str] | 否 | None | 路由模型的提示词，如果为 None 则使用默认的提示词 |

### 示例

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

---

## HandoffAgentMiddleware

用于实现多智能体切换（handoffs）的中间件。

### 类定义

```python
class HandoffAgentMiddleware(AgentMiddleware):
    state_schema = MultiAgentState
    def __init__(
        self,
        agents_config: dict[str, AgentConfig],
        custom_handoffs_tool_descriptions: Optional[dict[str, str]] = None,
    ) -> None:
```

### 参数

| 参数 | 类型 | 必填 | 默认值 | 描述 |
|------|------|------|--------|------|
| agents_config | dict[str, AgentConfig] | 是 | - | 智能体配置字典，键为智能体名称，值为智能体配置 |
| custom_handoffs_tool_descriptions | Optional[dict[str, str]] | 否 | None | 自定义交接到其它智能体的工具描述 |

### 示例

```python
handoffs_agent_middleware = HandoffsAgentMiddleware({
    "time_agent":{
        "model":"vllm:qwen3-4b",
        "prompt":"你是一个时间智能体，负责回答时间相关的问题。",
        "tools":[get_current_time, transfer_to_default_agent],
        "handoffs":["default_agent"]
    },
    "default_agent":{
        "model":"vllm:qwen3-8b",
        "prompt":"你是一个复杂任务智能体，负责回答复杂任务相关的问题。",
        "default":True,
        "handoffs":["time_agent"]
    }
})
```

---

## ToolCallRepairMiddleware

用于修复无效工具调用的中间件。

### 类定义

```python
class ToolCallRepairMiddleware(AgentMiddleware):
```

### 示例

```python
tool_call_repair_middleware = ToolCallRepairMiddleware()
```

---

## format_prompt

用于格式化提示词的中间件。

### 函数签名

```python
@dynamic_prompt
def format_prompt(request: ModelRequest) -> str
```

---

## PlanState

用于 Plan 的状态 Schema。

### 类定义

```python
class Plan(TypedDict):
    content: str
    status: Literal["pending", "in_progress", "done"]


class PlanState(AgentState):
    plan: NotRequired[list[Plan]]
```

### 属性

| 属性 | 类型 | 描述 |
|------|------|------|
| plan | NotRequired[list[Plan]] | 计划列表 |
| plan.content | str | 计划内容 |
| plan.status | Literal["pending", "in_progress", "done"] | 计划状态，取值为`pending`、`in_progress`、`done` |

---

## ModelDict

模型列表的类型。

### 类定义

```python
class ModelDict(TypedDict):
    model_name: str
    model_description: str
    tools: NotRequired[list[BaseTool | dict[str, Any]]]
    model_kwargs: NotRequired[dict[str, Any]]
    model_instance: NotRequired[BaseChatModel]
    model_system_prompt: NotRequired[str]
```

### 属性

| 属性 | 类型 | 必填 | 描述 |
|------|------|------|------|
| model_name | str | 是 | 模型名称 |
| model_description | str | 是 | 模型描述 |
| tools | NotRequired[list[BaseTool \| dict[str, Any]]] | 否 | 模型可用的工具 |
| model_kwargs | NotRequired[dict[str, Any]] | 否 | 传递给模型的额外参数 |
| model_instance | NotRequired[BaseChatModel] | 否 | 模型实例 |
| model_system_prompt | NotRequired[str] | 否 | 模型的系统提示词 |

---

## SelectModel

用于选择模型的工具类。

### 类定义

```python
class SelectModel(BaseModel):
    """Tool for model selection - Must call this tool to return the finally selected model"""

    model_name: str = Field(
        ...,
        description="Selected model name (must be the full model name, for example, openai:gpt-4o)",
    )
```

### 属性

| 属性 | 类型 | 必填 | 描述 |
|------|------|------|------|
| model_name | str | 是 | 选择的模型名称（必须是完整的模型名称，例如，openai:gpt-4o） |

---

## MultiAgentState

用于多智能体切换的状态 Schema。

### 类定义

```python
class MultiAgentState(AgentState):
    active_agent: NotRequired[str]
```

### 属性

| 属性 | 类型 | 描述 |
|------|------|------|
| active_agent | NotRequired[str] | 当前激活的智能体名称 |

---

## AgentConfig

智能体配置的类型。

### 类定义

```python
class AgentConfig(TypedDict):
    model: NotRequired[str | BaseChatModel]
    prompt: str | SystemMessage
    tools: NotRequired[list[BaseTool | dict[str, Any]]]
    default: NotRequired[bool]
    handoffs: list[str] | Literal["all"]
```

### 属性

| 属性 | 类型 | 必填 | 描述 |
|------|------|------|------|
| model | NotRequired[str \| BaseChatModel] | 否 | 模型名称或模型实例 |
| prompt | str \| SystemMessage | 是 | 智能体的提示词 |
| tools | list[BaseTool \| dict[str, Any]] | 是 | 智能体可用的工具 |
| default | NotRequired[bool] | 否 | 是否为默认智能体 |
| handoffs | list[str] \| Literal["all"] | 是 | 可以交接到的智能体名称列表，或"all"表示所有智能体 |


