# Agent Module API Reference Documentation

## create_agent

Creates an agent that provides the same functionality as the official LangChain `create_agent`, but with expanded string-based model specification.

### Function Signature

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

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| model | str | Yes | - | Model identifier string that can be loaded by `load_chat_model`. Can be specified in "provider:model-name" format |
| tools | Sequence[BaseTool \| Callable \| dict[str, Any]] \| None | No | None | List of tools available to the agent |
| system_prompt | str \| SystemMessage \| None | No | None | Custom system prompt for the agent |
| middleware | Sequence[AgentMiddleware[AgentState[ResponseT], ContextT]] | No | () | Middleware for the agent |
| response_format | ResponseFormat[ResponseT] \| type[ResponseT] \| None | No | None | Response format for the agent |
| state_schema | type[AgentState[ResponseT]] \| None | No | None | State schema for the agent |
| context_schema | type[ContextT] \| None | No | None | Context schema for the agent |
| checkpointer | Checkpointer \| None | No | None | Checkpoint for state persistence |
| store | BaseStore \| None | No | None | Store for data persistence |
| interrupt_before | list[str] \| None | No | None | Nodes to interrupt before execution |
| interrupt_after | list[str] \| None | No | None | Nodes to interrupt after execution |
| debug | bool | No | False | Enable debug mode |
| name | str \| None | No | None | Agent name |
| cache | BaseCache \| None | No | None | Cache |

### Notes

This function provides the same functionality as the official `langchain` `create_agent`, but with expanded model selection. The main difference is that the `model` parameter must be a string that can be loaded by the `load_chat_model` function, allowing for more flexible model selection using registered model providers.

### Example

```python
agent = create_agent(model="vllm:qwen3-4b", tools=[get_current_time])
```

---

## wrap_agent_as_tool

Wraps an agent as a tool.

### Function Signature

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

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| agent | CompiledStateGraph | Yes | - | Agent |
| tool_name | Optional[str] | No | None | Tool name |
| tool_description | Optional[str] | No | None | Tool description |
| pre_input_hooks | Optional[tuple[Callable[[str, ToolRuntime], str], Callable[[str, ToolRuntime], Awaitable[str]]] \| Callable[[str, ToolRuntime], str]] | No | None | Agent input preprocessing function |
| post_output_hooks | Optional[tuple[Callable[[str, list[AnyMessage], ToolRuntime], Any], Callable[[str, list[AnyMessage], ToolRuntime], Awaitable[Any]]] \| Callable[[str, list[AnyMessage], ToolRuntime], Any]] | No | None | Agent output postprocessing function |

### Example

```python
tool = wrap_agent_as_tool(agent)
```

---

## create_write_plan_tool

Creates a tool for writing or updating plans.

### Function Signature

```python
def create_write_plan_tool(
    description: Optional[str] = None,
    message_key: Optional[str] = None,
) -> BaseTool
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| description | Optional[str] | No | None | Tool description |
| message_key | Optional[str] | No | None | Key for updating messages, defaults to "messages" |

### Example

```python
write_plan_tool = create_write_plan_tool()
```

---

## create_finish_sub_plan_tool

Creates a tool for updating execution status after completing subtasks.

### Function Signature

```python
def create_finish_sub_plan_tool(
    description: Optional[str] = None,
    message_key: Optional[str] = None,
) -> BaseTool
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| description | Optional[str] | No | None | Tool description |
| message_key | Optional[str] | No | None | Key for updating messages, defaults to "messages" |

### Example

```python
finish_sub_plan_tool = create_finish_sub_plan_tool()
```

---

## create_read_plan_tool

Creates a tool for reading execution status.

### Function Signature

```python
def create_read_plan_tool(
    description: Optional[str] = None,
) -> BaseTool
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| description | Optional[str] | No | None | Tool description |

### Example

```python
read_plan_tool = create_read_plan_tool()
```

---

## SummarizationMiddleware

Middleware for agent context summarization.

### Class Definition

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

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| model | str | Yes | - | Model identifier string that can be loaded by `load_chat_model`. Can be specified in "provider:model-name" format |
| trigger | ContextSize \| list[ContextSize] \| None | No | None | Context size that triggers summarization |
| keep | ContextSize | No | ("messages", _DEFAULT_MESSAGES_TO_KEEP) | Context size to keep |
| token_counter | TokenCounter | No | count_tokens_approximately | Token counter |
| summary_prompt | str | No | DEFAULT_SUMMARY_PROMPT | Summary prompt |
| trim_tokens_to_summarize | int \| None | No | _DEFAULT_TRIM_TOKEN_LIMIT | Number of tokens to trim before summarization |

### Example

```python
summarization_middleware = SummarizationMiddleware(model="vllm:qwen3-4b")
```

---

## LLMToolSelectorMiddleware

Middleware for agent tool selection.

### Class Definition

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

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| model | str | Yes | - | Model identifier string that can be loaded by `load_chat_model`. Can be specified in "provider:model-name" format |
| system_prompt | Optional[str] | No | None | System prompt |
| max_tools | Optional[int] | No | None | Maximum number of tools |
| always_include | Optional[list[str]] | No | None | Tools to always include |

### Example

```python
llm_tool_selector_middleware = LLMToolSelectorMiddleware(model="vllm:qwen3-4b")
```

---

## PlanMiddleware

Middleware for agent plan management.

### Class Definition

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
    ) -> None
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| system_prompt | Optional[str] | No | None | System prompt |
| write_plan_tool_description | Optional[str] | No | None | Description of the write plan tool |
| finish_sub_plan_tool_description | Optional[str] | No | None | Description of the finish sub plan tool |
| read_plan_tool_description | Optional[str] | No | None | Description of the read plan tool |
| use_read_plan_tool | bool | No | True | Whether to use the read plan tool |

### Example

```python
plan_middleware = PlanMiddleware()
```

---

## ModelFallbackMiddleware

Middleware for agent model fallback.

### Class Definition

```python
class ModelFallbackMiddleware(_ModelFallbackMiddleware):
    def __init__(
        self,
        first_model: str,
        *additional_models: str,
    ) -> None
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| first_model | str | Yes | - | Model identifier string that can be loaded by `load_chat_model`. Can be specified in "provider:model-name" format |
| additional_models | str | No | - | List of backup models |

### Example

```python
model_fallback_middleware = ModelFallbackMiddleware(
    "vllm:qwen3-4b",
    "vllm:qwen3-8b"
)
```

---

## LLMToolEmulator

Middleware for using large models to simulate tool calls.

### Class Definition

```python
class LLMToolEmulator(_LLMToolEmulator):
    def __init__(
        self,
        *,
        model: str,
        tools: list[str | BaseTool] | None = None,
    ) -> None
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| model | str | Yes | - | Model identifier string that can be loaded by `load_chat_model`. Can be specified in "provider:model-name" format |
| tools | list[str \| BaseTool] \| None | No | None | List of tools |

### Example

```python
llm_tool_emulator = LLMToolEmulator(model="vllm:qwen3-4b", tools=[get_current_time])
```

---

## ModelRouterMiddleware

Middleware for dynamically routing to appropriate models based on input content.

### Class Definition

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

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| router_model | str \| BaseChatModel | Yes | - | Model for routing, accepts string type (loaded with `load_chat_model`) or directly passed ChatModel |
| model_list | list[ModelDict] | Yes | - | List of models, each model needs to contain `model_name` and `model_description` keys, and can optionally include `tools`, `model_kwargs`, `model_instance`, `model_system_prompt` keys |
| router_prompt | Optional[str] | No | None | Prompt for the routing model, if None uses the default prompt |

### Example

```python
model_router_middleware = ModelRouterMiddleware(
    router_model="vllm:qwen3-4b",
    model_list=[
        {
            "model_name": "vllm:qwen3-4b",
            "model_description": "Suitable for ordinary tasks such as dialogue, text generation, etc."
        },
        {
            "model_name": "vllm:qwen3-8b",
            "model_description": "Suitable for complex tasks such as code generation, data analysis, etc.",
        },
    ]
)
```

---

## ToolCallRepairMiddleware

Middleware for repairing invalid tool calls.

### Class Definition

```python
class ToolCallRepairMiddleware(AgentMiddleware):
```

### Example

```python
tool_call_repair_middleware = ToolCallRepairMiddleware()
```

---

## format_prompt

Middleware for formatting prompts.

### Function Signature

```python
@dynamic_prompt
def format_prompt(request: ModelRequest) -> str
```

---

## PlanState

State Schema for Plan.

### Class Definition

```python
class Plan(TypedDict):
    content: str
    status: Literal["pending", "in_progress", "done"]


class PlanState(AgentState):
    plan: NotRequired[list[Plan]]
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| plan | NotRequired[list[Plan]] | List of plans |
| plan.content | str | Plan content |
| plan.status | Literal["pending", "in_progress", "done"] | Plan status, values are `pending`, `in_progress`, `done` |

---

## ModelDict

Type for model list.

### Class Definition

```python
class ModelDict(TypedDict):
    model_name: str
    model_description: str
    tools: NotRequired[list[BaseTool | dict[str, Any]]]
    model_kwargs: NotRequired[dict[str, Any]]
    model_instance: NotRequired[BaseChatModel]
    model_system_prompt: NotRequired[str]
```

### Properties

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| model_name | str | Yes | Model name |
| model_description | str | Yes | Model description |
| tools | NotRequired[list[BaseTool \| dict[str, Any]]] | No | Tools available to the model |
| model_kwargs | NotRequired[dict[str, Any]] | No | Additional parameters to pass to the model |
| model_instance | NotRequired[BaseChatModel] | No | Model instance |
| model_system_prompt | NotRequired[str] | No | System prompt for the model |

---

## SelectModel

Tool class for model selection.

### Class Definition

```python
class SelectModel(BaseModel):
    """Tool for model selection - Must call this tool to return the finally selected model"""

    model_name: str = Field(
        ...,
        description="Selected model name (must be the full model name, for example, openai:gpt-4o)",
    )
```

### Properties

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| model_name | str | Yes | Selected model name (must be the full model name, for example, openai:gpt-4o) |