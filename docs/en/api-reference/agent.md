# Agent Module API Reference

## create_agent

Creates an agent with the same functionality as the official `langchain` `create_agent`, but extends the model specification to a string.

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
| middleware | Sequence[AgentMiddleware[AgentState[ResponseT], ContextT]] | No | () | Agent middleware |
| response_format | ResponseFormat[ResponseT] \| type[ResponseT] \| None | No | None | Response format for the agent |
| state_schema | type[AgentState[ResponseT]] \| None | No | None | State schema for the agent |
| context_schema | type[ContextT] \| None | No | None | Context schema for the agent |
| checkpointer | Checkpointer \| None | No | None | Checkpointer for state persistence |
| store | BaseStore \| None | No | None | Storage for data persistence |
| interrupt_before | list[str] \| None | No | None | Nodes to interrupt before execution |
| interrupt_after | list[str] \| None | No | None | Nodes to interrupt after execution |
| debug | bool | No | False | Enable debug mode |
| name | str \| None | No | None | Agent name |
| cache | BaseCache \| None | No | None | Cache |

### Notes

This function provides the same functionality as the official `langchain` `create_agent`, but extends the model selection. The main difference is that the `model` parameter must be a string that can be loaded by the `load_chat_model` function, allowing more flexible model selection using registered model providers.

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
            Callable[[str, ToolRuntime], str | dict[str, Any]],
            Callable[[str, ToolRuntime], Awaitable[str | dict[str, Any]]],
        ]
        | Callable[[str, ToolRuntime], str | dict[str, Any]]
    ] = None,
    post_output_hooks: Optional[
        tuple[
            Callable[[str, dict[str, Any], ToolRuntime], Any],
            Callable[[str, dict[str, Any], ToolRuntime], Awaitable[Any]],
        ]
        | Callable[[str, dict[str, Any], ToolRuntime], Any]
    ] = None,
) -> BaseTool:
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| agent | CompiledStateGraph | Yes | - | The agent graph to wrap |
| tool_name | Optional[str] | No | None | Name of the resulting tool |
| tool_description | Optional[str] | No | None | Description of the resulting tool |
| pre_input_hooks | - | No | None | Hooks for processing input before passing to the agent |
| post_output_hooks | - | No | None | Hooks for processing output after receiving from the agent |

### Example

```python
tool = wrap_agent_as_tool(agent)
```

---

## wrap_all_agents_as_tool

Wraps all agents as a single tool.

### Function Signature

```python
def wrap_all_agents_as_tool(
    agents: list[CompiledStateGraph],
    tool_name: Optional[str] = None,
    tool_description: Optional[str] = None,
    pre_input_hooks: Optional[
        tuple[
            Callable[[str, ToolRuntime], str | dict[str, Any]],
            Callable[[str, ToolRuntime], Awaitable[str | dict[str, Any]]],
        ]
        | Callable[[str, ToolRuntime], str | dict[str, Any]]
    ] = None,
    post_output_hooks: Optional[
        tuple[
            Callable[[str, dict[str, Any], ToolRuntime], Any],
            Callable[[str, dict[str, Any], ToolRuntime], Awaitable[Any]],
        ]
        | Callable[[str, dict[str, Any], ToolRuntime], Any]
    ] = None,
) -> BaseTool:
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| agents | list[CompiledStateGraph] | Yes | - | List of agents (must contain at least 2 agents, and each agent must have a unique name) |
| tool_name | Optional[str] | No | None | Name of the resulting tool |
| tool_description | Optional[str] | No | None | Description of the resulting tool |
| pre_input_hooks | - | No | None | Hooks for processing input before passing to the agents |
| post_output_hooks | - | No | None | Hooks for processing output after receiving from the agents |

### Example

```python
tool = wrap_all_agents_as_tool([time_agent, weather_agent])
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
| trigger | ContextSize \| list[ContextSize] \| None | No | None | Context size threshold that triggers summarization |
| keep | ContextSize | No | ("messages", _DEFAULT_MESSAGES_TO_KEEP) | Context size to preserve after summarization |
| token_counter | TokenCounter | No | count_tokens_approximately | Token counting function to use |
| summary_prompt | str | No | DEFAULT_SUMMARY_PROMPT | System prompt used for summarization |
| trim_tokens_to_summarize | int \| None | No | _DEFAULT_TRIM_TOKEN_LIMIT | Number of tokens to trim from the context before summarizing |

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
| system_prompt | Optional[str] | No | None | System prompt for the selection model |
| max_tools | Optional[int] | No | None | Maximum number of tools to select |
| always_include | Optional[list[str]] | No | None | List of tool names to always include in the selection |

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
        custom_plan_tool_descriptions: Optional[PlanToolDescription] = None,
        use_read_plan_tool: bool = True,
    ) -> None
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| system_prompt | Optional[str] | No | None | System prompt for the planning agent |
| custom_plan_tool_descriptions | Optional[PlanToolDescription] | No | None | Custom descriptions for plan-related tools |
| use_read_plan_tool | bool | No | True | Whether to enable the tool that allows the model to read the current plan |

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
| first_model | str | Yes | - | Primary model identifier string (can be loaded by `load_chat_model`) |
| additional_models | str | No | - | Fallback model identifier strings to use if the primary model fails |

### Example

```python
model_fallback_middleware = ModelFallbackMiddleware(
    "vllm:qwen3-4b",
    "vllm:qwen3-8b"
)
```

---

## LLMToolEmulator

Middleware for simulating tool calls using large language models.

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
| tools | list[str \| BaseTool] \| None | No | None | List of tools to be emulated by the LLM |

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
| router_model | str \| BaseChatModel | Yes | - | Model used for routing. Accepts a string (loaded via `load_chat_model`) or a ChatModel instance |
| model_list | list[ModelDict] | Yes | - | List of available models. Each entry must contain `model_name` and `model_description`, and optionally `tools`, `model_kwargs`, `model_instance`, and `model_system_prompt` |
| router_prompt | Optional[str] | No | None | Custom prompt for the router model. Uses default prompt if not provided |

### Example

```python
model_router_middleware = ModelRouterMiddleware(
    router_model="vllm:qwen3-4b",
    model_list=[
        {
            "model_name": "vllm:qwen3-4b",
            "model_description": "Suitable for general tasks such as conversation, text generation, etc."
        },
        {
            "model_name": "vllm:qwen3-8b",
            "model_description": "Suitable for complex tasks such as code generation, data analysis, etc.",
        },
    ]
)
```

---

## HandoffAgentMiddleware

Middleware for implementing multi-agent handoffs.

### Class Definition

```python
class HandoffAgentMiddleware(AgentMiddleware):
    state_schema = MultiAgentState
    def __init__(
        self,
        agents_config: dict[str, AgentConfig],
        custom_handoffs_tool_descriptions: Optional[dict[str, str]] = None,
    ) -> None
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| agents_config | dict[str, AgentConfig] | Yes | - | Configuration dictionary for agents. Keys are agent names, values are agent configurations |
| custom_handoffs_tool_descriptions | Optional[dict[str, str]] | No | None | Custom descriptions for handoff tools targeting other agents |

### Example

```python
handoffs_agent_middleware = HandoffsAgentMiddleware({
    "time_agent":{
        "model":"vllm:qwen3-4b",
        "prompt":"You are a time agent responsible for answering time-related questions.",
        "tools":[get_current_time, transfer_to_default_agent],
        "handoffs":["default_agent"]
    },
    "default_agent":{
        "model":"vllm:qwen3-8b",
        "prompt":"You are a complex task agent responsible for answering complex task-related questions.",
        "default":True,
        "handoffs":["time_agent"]
    }
})
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

Helper for formatting prompts.

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

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| plan | NotRequired[list[Plan]] | List of plan steps |
| plan.content | str | Content of the plan step |
| plan.status | Literal["pending", "in_progress", "done"] | Status of the plan step. Valid values are `pending`, `in_progress`, `done` |

---

## ModelDict

Type definition for the model list.

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

### Attributes

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| model_name | str | Yes | Name of the model |
| model_description | str | Yes | Description of the model's capabilities |
| tools | NotRequired[list[BaseTool \| dict[str, Any]]] | No | Tools available to this model |
| model_kwargs | NotRequired[dict[str, Any]] | No | Additional keyword arguments to pass to the model |
| model_instance | NotRequired[BaseChatModel] | No | A specific model instance to use |
| model_system_prompt | NotRequired[str] | No | System prompt specific to this model |

---

## SelectModel

Tool class for selecting a model.

### Class Definition

```python
class SelectModel(BaseModel):
    """Tool for model selection - Must call this tool to return the finally selected model"""

    model_name: str = Field(
        ...,
        description="Selected model name (must be the full model name, for example, openai:gpt-4o)",
    )
```

### Attributes

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| model_name | str | Yes | The name of the selected model (must be the full model name, e.g., openai:gpt-4o) |

---

## MultiAgentState

State Schema for multi-agent handoffs.

### Class Definition

```python
class MultiAgentState(AgentState):
    active_agent: NotRequired[str]
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| active_agent | NotRequired[str] | The name of the currently active agent |

---

## AgentConfig

Type definition for agent configuration.

### Class Definition

```python
class AgentConfig(TypedDict):
    model: NotRequired[str | BaseChatModel]
    prompt: str | SystemMessage
    tools: NotRequired[list[BaseTool | dict[str, Any]]]
    default: NotRequired[bool]
    handoffs: list[str] | Literal["all"]
```

### Attributes

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| model | NotRequired[str \| BaseChatModel] | No | Model name or model instance |
| prompt | str \| SystemMessage | Yes | System prompt for the agent |
| tools | list[BaseTool \| dict[str, Any]] | Yes | Tools available to the agent |
| default | NotRequired[bool] | No | Whether this agent is the default fallback |
| handoffs | list[str] \| Literal["all"] | Yes | List of agent names to hand off to, or "all" to allow handoffs to any agent |