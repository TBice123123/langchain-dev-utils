# Agent Module API Reference

## create_agent

A pre-built agent function that provides the same functionality as LangChain's official `create_agent`, but with the extension of specifying models with strings.

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

**Parameter Description:**

- `model`: String type, required, a model identifier string that can be loaded by `load_chat_model`. Can be specified in "provider:model-name" format
- `tools`: Sequence of BaseTool, Callable, or dictionary, or ToolNode type, required, list of tools available to the agent
- `system_prompt`: Optional string type, custom system prompt for the agent
- `middleware`: Optional AgentMiddleware type, middleware for the agent
- `response_format`: Optional ResponseFormat type, response format for the agent
- `state_schema`: Optional StateSchemaType type, state schema for the agent
- `context_schema`: Optional any type, context schema for the agent
- `checkpointer`: Optional Checkpointer type, checkpoint for state persistence
- `store`: Optional BaseStore type, storage for data persistence
- `interrupt_before`: Optional list of strings, nodes to interrupt before execution
- `interrupt_after`: Optional list of strings, nodes to interrupt after execution
- `debug`: Boolean type, optional, enable debug mode, default is False
- `name`: Optional string type, agent name
- `cache`: Optional BaseCache type, cache

**Return Value:** CompiledStateGraph type

**Note:** This function provides the same functionality as LangChain's official `create_agent`, but with extended model selection. The main difference is that the `model` parameter must be a string that can be loaded by the `load_chat_model` function, allowing for more flexible model selection using registered model providers.

**Example:**

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

**Parameter Description:**

- `agent`: CompiledStateGraph type, required, the agent
- `tool_name`: Optional string type, tool name
- `tool_description`: Optional string type, tool description
- `pre_input_hooks`: Optional tuple type or function type, Agent input preprocessing function
- `post_output_hooks`: Optional tuple type or function type, Agent output post-processing function

**Return Value:** BaseTool type, tool instance

**Example:**

```python
tool = wrap_agent_as_tool(agent)
```

## create_write_plan_tool

Creates a tool for writing or updating plans.

```python
def create_write_plan_tool(
    description: Optional[str] = None,
    message_key: Optional[str] = None,
) -> BaseTool:
```

**Parameter Description:**

- `description`: Optional string type, tool description
- `message_key`: Optional string type, key for updating messages, default "messages"

**Return Value:** BaseTool type, tool instance

**Example:**

```python
write_plan_tool = create_write_plan_tool()
```

## create_finish_sub_plan_tool

Creates a tool for updating execution status after completing subtasks.

```python
def create_finish_sub_plan_tool(
    description: Optional[str] = None,
    message_key: Optional[str] = None,
) -> BaseTool:
```

**Parameter Description:**

- `description`: Optional string type, tool description
- `message_key`: Optional string type, key for updating messages, default "messages"

**Return Value:** BaseTool type, tool instance

**Example:**

```python
finish_sub_plan_tool = create_finish_sub_plan_tool()
```

## create_read_plan_tool

Creates a tool for reading execution status.

```python
def create_read_plan_tool(
    description: Optional[str] = None,
) -> BaseTool:
```

**Parameter Description:**

- `description`: Optional string type, tool description

**Return Value:** BaseTool type, tool instance

**Example:**

```python
read_plan_tool = create_read_plan_tool()
```

## SummarizationMiddleware

Middleware for agent context summarization.

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

**Parameter Description:**

- `model`: String type, required, a model identifier string that can be loaded by `load_chat_model`. Can be specified in "provider:model-name" format
- `trigger`: Optional ContextSize type or list type, context size that triggers summarization
- `keep`: Optional ContextSize type, context size to keep
- `token_counter`: Optional TokenCounter type, token counter
- `summary_prompt`: Optional string type, summarization prompt
- `trim_tokens_to_summarize`: Optional integer type, number of tokens to trim before summarization

**Example:**

```python
summarization_middleware = SummarizationMiddleware(model="vllm:qwen3-4b")
```

## LLMToolSelectorMiddleware

Middleware for agent tool selection.

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

**Parameter Description:**

- `model`: String type, required, a model identifier string that can be loaded by `load_chat_model`. Can be specified in "provider:model-name" format
- `system_prompt`: Optional string type, system prompt
- `max_tools`: Optional integer type, maximum number of tools
- `always_include`: Optional list of strings, tools to always include

**Example:**

```python
llm_tool_selector_middleware = LLMToolSelectorMiddleware(model="vllm:qwen3-4b")
```

## PlanMiddleware

Middleware for agent plan management.

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

**Parameter Description:**

- `system_prompt`: Optional string type, system prompt
- `write_plan_tool_description`: Optional string type, description of the write plan tool
- `finish_sub_plan_tool_description`: Optional string type, description of the finish sub plan tool
- `read_plan_tool_description`: Optional string type, description of the read plan tool
- `use_read_plan_tool`: Optional boolean type, whether to use the read plan tool

**Example:**

```python
plan_middleware = PlanMiddleware()
```

## ModelFallbackMiddleware

Middleware for agent model fallback.

```python
class ModelFallbackMiddleware(_ModelFallbackMiddleware):
    def __init__(
        self,
        first_model: str,
        *additional_models: str,
    ) -> None:
```

**Parameter Description:**

- `first_model`: String type, required, a model identifier string that can be loaded by `load_chat_model`. Can be specified in "provider:model-name" format
- `additional_models`: Optional list of strings, backup model list

**Example:**

```python
model_fallback_middleware = ModelFallbackMiddleware(
    "vllm:qwen3-4b",
    "vllm:qwen3-8b"
)
```

## LLMToolEmulator

Middleware for using large models to simulate tool calls.

```python
class LLMToolEmulator(_LLMToolEmulator):
    def __init__(
        self,
        *,
        model: str,
        tools: list[str | BaseTool] | None = None,
    ) -> None:
```

**Parameter Description:**

- `model`: String type, required, a model identifier string that can be loaded by `load_chat_model`. Can be specified in "provider:model-name" format
- `tools`: Optional list of BaseTool type, tool list

**Example:**

```python
llm_tool_emulator = LLMToolEmulator(model="vllm:qwen3-4b", tools=[get_current_time])
```

## ModelRouterMiddleware

Middleware for dynamically routing to appropriate models based on input content.

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

**Parameter Description:**

- `router_model`: Model for routing, accepts string type (loaded with `load_chat_model`) or directly passed ChatModel
- `model_list`: List of models, each model needs to contain two keys: `model_name` and `model_description`, and can optionally include four keys: `tools`, `model_kwargs`, `model_instance`, and `model_system_prompt`, representing the tools available to the model (if not passed, defaults to using all tools), additional parameters passed to the model (e.g., temperature, top_p, etc.), the model instance corresponding to that model, and the system prompt for that model.
- `router_prompt`: Prompt for the routing model, if None, the default prompt is used

**Example:**

```python
model_router_middleware = ModelRouterMiddleware(
    router_model="vllm:qwen3-4b",
    model_list=[
        {
            "model_name": "vllm:qwen3-4b",
            "model_description": "Suitable for ordinary tasks, such as dialogue, text generation, etc."
        },
        {
            "model_name": "vllm:qwen3-8b",
            "model_description": "Suitable for complex tasks, such as code generation, data analysis, etc.",
        },
    ]
)
```

## ToolCallRepairMiddleware

Middleware for repairing invalid tool calls.

```python
class ToolCallRepairMiddleware(AgentMiddleware):
```

**Example:**

```python
tool_call_repair_middleware = ToolCallRepairMiddleware()
```

## format_prompt

Middleware for formatting prompts.

```python
@dynamic_prompt
def format_prompt(request: ModelRequest) -> str:
```



## PlanState

State Schema for Plan.

```python
class Plan(TypedDict):
    content: str
    status: Literal["pending", "in_progress", "done"]


class PlanState(AgentState):
    plan: NotRequired[list[Plan]]
```

- `plan`: Optional list type, plan list
- `plan.content`: Plan content
- `plan.status`: Plan status, values are `pending`, `in_progress`, `done`

## ModelDict

Type for model list.

```python
class ModelDict(TypedDict):
    model_name: str
    model_description: str
    tools: NotRequired[list[BaseTool | dict[str, Any]]]
    model_kwargs: NotRequired[dict[str, Any]]
    model_instance: NotRequired[BaseChatModel]
    model_system_prompt: NotRequired[str]
```

## SelectModel

Tool class for model selection.

```python
class SelectModel(BaseModel):
    """Tool for model selection - Must call this tool to return the finally selected model"""

    model_name: str = Field(
        ...,
        description="Selected model name (must be the full model name, for example, openai:gpt-4o)",
    )
```