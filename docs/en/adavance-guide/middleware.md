# Middleware

## Overview

Middleware is a component specifically built for LangChain's pre-built Agents. The official library provides some built-in middleware, and this library offers more practical middleware based on actual usage scenarios.

The middleware provided by this library includes:

- `PlanMiddleware`: Task planning that breaks down complex tasks into ordered subtasks
- `ModelRouterMiddleware`: Dynamically routes to the most suitable model based on input content
- `HandoffAgentMiddleware`: Flexibly switches between multiple sub-agents
- `ToolCallRepairMiddleware`: Automatically fixes invalid tool calls from large models
- `format_prompt`: Dynamically formats placeholders in system prompts

Additionally, this library extends the functionality of official middleware, supporting model specification through string parameters:

- SummarizationMiddleware
- LLMToolSelectorMiddleware
- ModelFallbackMiddleware
- LLMToolEmulator

## Task Planning

`PlanMiddleware` is a middleware for structured decomposition and process management before executing complex tasks.

!!! info "Additional Information"

    Task planning is an efficient context engineering management strategy. Before executing a task, the large model first breaks down the overall task into multiple ordered subtasks, forming a task planning list (called a "plan" in this library). It then executes each subtask in sequence and dynamically updates the task status after completing each step until all subtasks are finished.

### Parameter Description

| Parameter | Description |
|-----------|-------------|
| `system_prompt` | System prompt. If `None`, the default prompt is used.<br><br>**Type**: `str`<br>**Required**: No |
| `custom_plan_tool_descriptions` | Custom descriptions for plan-related tools.<br><br>**Type**: `dict`<br>**Required**: No |
| `use_read_plan_tool` | Whether to enable the read plan tool.<br><br>**Type**: `bool`<br>**Required**: No<br>**Default**: `True` |

The keys in the `custom_plan_tool_descriptions` dictionary can take the following three values:

| Key | Description |
|-----|-------------|
| `write_plan` | Description of the write plan tool |
| `finish_sub_plan` | Description of the finish subplan tool |
| `read_plan` | Description of the read plan tool |

### Usage Example

```python
from langchain_dev_utils.agents.middleware import PlanMiddleware

agent = create_agent(
    model="vllm:qwen3-4b",
    middleware=[
        PlanMiddleware(
            custom_plan_tool_descriptions={
                "write_plan": "Used for writing plans, breaking tasks into multiple ordered subtasks.",
                "finish_sub_plan": "Used for completing subtasks, updating subtask status to completed.",
                "read_plan": "Used for querying the current task planning list."
            },
            use_read_plan_tool=True,  # If not using the read plan tool, set this parameter to False
        )
    ],
)

response = agent.invoke(
    {"messages": [HumanMessage(content="I want to visit New York for a few days, help me plan my itinerary")]}
)
print(response)
```

### Tool Description

`PlanMiddleware` requires the use of two tools: `write_plan` and `finish_sub_plan`, while the `read_plan` tool is enabled by default; if not needed, the `use_read_plan_tool` parameter can be set to `False`.

### Comparison with Official To-do List Middleware

This middleware has a similar functional positioning to LangChain's official **To-do list middleware**, but there are differences in tool design:

| Feature | Official To-do List Middleware | This Library's PlanMiddleware |
|---------|-------------------------------|-------------------------------|
| Number of Tools | 1 (`write_todo`) | 3 (`write_plan`, `finish_sub_plan`, `read_plan`) |
| Functional Positioning | Focused on to-do lists | Specifically for planning lists |
| Operation Method | Adding and modifying done through one tool | Writing, modifying, and querying done through different tools |

Whether it's `todo` or `plan`, they are essentially the same concept. The key difference between this middleware and the official one is that it provides three specialized tools:

- `write_plan`: Used for writing or updating plan content
- `finish_sub_plan`: Used to update the status after completing a subtask
- `read_plan`: Used for querying plan content

## Model Routing

`ModelRouterMiddleware` is a middleware for **dynamically routing to the most suitable model based on input content**. It analyzes user requests through a "routing model" and selects the most appropriate model from a predefined list to handle the current task.

### Parameter Description

| Parameter | Description |
|-----------|-------------|
| `router_model` | Model used for routing decisions.<br><br>**Type**: `str` \| `BaseChatModel`<br>**Required**: Yes |
| `model_list` | List of model configurations.<br><br>**Type**: `list[ModelDict]`<br>**Required**: Yes |
| `router_prompt` | Custom prompt for the routing model.<br><br>**Type**: `str`<br>**Required**: No |

#### `model_list` Configuration Description

Each model configuration is a dictionary containing the following fields:

| Field | Description |
|-------|-------------|
| `model_name` | Unique identifier for the model, using `provider:model-name` format.<br><br>**Type**: `str`<br>**Required**: Yes |
| `model_description` | Brief description of the model's capabilities or applicable scenarios.<br><br>**Type**: `str`<br>**Required**: Yes |
| `tools` | Whitelist of tools that this model can call.<br><br>**Type**: `list[BaseTool]`<br>**Required**: No |
| `model_kwargs` | Additional parameters when loading the model.<br><br>**Type**: `dict`<br>**Required**: No |
| `model_system_prompt` | System-level prompt for the model.<br><br>**Type**: `str`<br>**Required**: No |
| `model_instance` | Instantiated model object.<br><br>**Type**: `BaseChatModel`<br>**Required**: No |

!!! tip "model_instance Field Description"

    - **If provided**: Directly uses this instance, `model_name` is only for identification, `model_kwargs` is ignored; suitable for cases not using this library's conversation model management functionality.
    - **If not provided**: Loads the model using `load_chat_model` based on `model_name` and `model_kwargs`.
    - **Naming format**: In either case, it's recommended to use the `provider:model-name` format for `model_name`.

### Usage Example

#### Step 1: Define the Model List

```python
from langchain_dev_utils.agents.middleware.model_router import ModelDict

model_list: list[ModelDict] = [
    {
        "model_name": "vllm:qwen3-8b",
        "model_description": "Suitable for ordinary tasks, such as dialogue, text generation, etc.",
        "model_kwargs": {
            "temperature": 0.7,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
        "model_system_prompt": "You are an assistant, good at handling ordinary tasks, such as dialogue, text generation, etc.",
    },
    {
        "model_name": "vllm:qwen3-vl-2b",
        "model_description": "Suitable for visual tasks",
        "tools": [],  # If this model doesn't need any tools, set this field to an empty list []
    },
    {
        "model_name": "vllm:qwen3-coder-flash",
        "model_description": "Suitable for code generation tasks",
        "tools": [run_python_code],  # Only allow the use of run_python_code tool
    },
    {
        "model_name": "openai:gpt-4o",
        "model_description": "Suitable for comprehensive high-difficulty tasks",
        "model_system_prompt": "You are an assistant, good at handling comprehensive high-difficulty tasks",
        "model_instance": ChatOpenAI(
            model_name="gpt-4o"
        ),  # Directly pass the instance, where model_name is only for identification, model_kwargs is ignored
    },
]
```

#### Step 2: Create Agent and Enable Middleware

```python
from langchain_dev_utils.agents.middleware import ModelRouterMiddleware
from langchain_core.messages import HumanMessage

agent = create_agent(
    model="vllm:qwen3-4b",  # This model is only a placeholder, actually dynamically replaced by middleware
    tools=[run_python_code, get_current_time],
    middleware=[
        ModelRouterMiddleware(
            router_model="vllm:qwen3-4b",
            model_list=model_list,
        )
    ],
)

# The routing middleware will automatically select the most suitable model based on input content
response = agent.invoke({"messages": [HumanMessage(content="Help me write a bubble sort code")]})
print(response)
```

Through `ModelRouterMiddleware`, you can easily build a multi-model, multi-capability Agent that automatically selects the optimal model based on task type, improving response quality and efficiency.

## Agent Handoffs

`HandoffAgentMiddleware` is a middleware used for **flexibly switching between multiple sub-agents**, fully implementing LangChain's official `handoffs` multi-agent collaboration scheme.

### Parameter Description

| Parameter | Description |
|------|------|
| `agents_config` | A dictionary of agent configurations, where the key is the agent name and the value is the agent configuration dictionary.<br><br>**Type**: `dict[str, AgentConfig]`<br>**Required**: Yes |
| `custom_handoffs_tool_descriptions` | Custom descriptions for handoff tools, where the key is the agent name and the value is the corresponding handoff tool description.<br><br>**Type**: `dict[str, str]`<br>**Required**: No |
| `handoffs_tool_overrides` | Custom implementations of handoff tools, where the key is the agent name and the value is the corresponding handoff tool implementation.<br><br>**Type**: `dict[str, BaseTool]`<br>**Required**: No |

#### `agents_config` Configuration Description

Each agent is configured as a dictionary containing the following fields:

| Field | Description |
|------|------|
| `model` | Specifies the model used by this agent; if not passed, it inherits the model corresponding to the `model` parameter of `create_agent`. Supports strings (must be in `provider:model-name` format, e.g., `vllm:qwen3-4b`) or a `BaseChatModel` instance.<br><br>**Type**: `str` \| `BaseChatModel`<br>**Required**: No |
| `prompt` | The system prompt for the agent.<br><br>**Type**: `str` \| `SystemMessage`<br>**Required**: Yes |
| `tools` | A list of tools that the agent can call.<br><br>**Type**: `list[BaseTool]`<br>**Required**: No |
| `default` | Whether to set as the default agent; defaults to `False`. There must be exactly one agent set to `True` in the entire configuration.<br><br>**Type**: `bool`<br>**Required**: No |
| `handoffs` | A list of other agent names that this agent can hand off to. If set to `"all"`, it means this agent can hand off to all other agents.<br><br>**Type**: `list[str]` \| `str`<br>**Required**: Yes |

For this paradigm of multi-agent implementation, a tool used for handoffs is often required. This middleware utilizes the `handoffs` configuration of each agent to automatically create the corresponding handoff tool for each agent. If you wish to customize the description of the handoff tool, you can achieve this via the `custom_handoffs_tool_descriptions` parameter.


**Usage Example**

In this example, we will use four agents: `time_agent`, `weather_agent`, `code_agent`, and `default_agent`.

Next, we need to create the corresponding agent configuration dictionary `agent_config`.

```python
from langchain_dev_utils.agents.middleware.handoffs import AgentConfig

agent_config: dict[str, AgentConfig] = {
    "time_agent": {
        "model": "vllm:qwen3-8b",
        "prompt": "You are a time assistant",
        "tools": [get_current_time],
        "handoffs": ["default_agent"],  # This agent can only hand off to default_agent
    },
    "weather_agent": {
        "prompt": "You are a weather assistant",
        "tools": [get_current_weather, get_current_city],
        "handoffs": ["default_agent"],
    },
    "code_agent": {
        "model": load_chat_model("vllm:qwen3-coder-flash"),
        "prompt": "You are a coding assistant",
        "tools": [
            run_code,
        ],
        "handoffs": ["default_agent"],
    },
    "default_agent": {
        "prompt": "You are an assistant",
        "default": True, # Set as the default agent
        "handoffs": "all",  # This agent can hand off to all other agents
    },
}
```

Finally, simply pass this configuration to `HandoffAgentMiddleware`.

```python
from langchain_dev_utils.agents.middleware import HandoffAgentMiddleware

agent = create_agent(
    model="vllm:qwen3-4b",
    tools=[
        get_current_time,
        get_current_weather,
        get_current_city,
        run_code,
    ],
    middleware=[HandoffAgentMiddleware(agents_config=agent_config)],
)

response = agent.invoke({"messages": [HumanMessage(content="What is the current time?")]})
print(response)
```

If you want to customize the description of the handoff tools, you can pass the second parameter `custom_handoffs_tool_descriptions`.

```python
from langchain_dev_utils.agents.middleware import HandoffAgentMiddleware

agent = create_agent(
    model="vllm:qwen3-4b",
    tools=[
        get_current_time,
        get_current_weather,
        get_current_city,
        run_code,
    ],
    middleware=[
        HandoffAgentMiddleware(
            agents_config=agent_config,
            custom_handoffs_tool_descriptions={
                "time_agent": "Use this tool to hand off to the time assistant to resolve time query issues",
                "weather_agent": "Use this tool to hand off to the weather assistant to resolve weather query issues",
                "code_agent": "Use this tool to hand off to the code assistant to resolve coding issues",
                "default_agent": "Use this tool to hand off to the default assistant",
            },
        )
    ],
)
```

If you want to completely customize the logic of the handoff tool implementation, you can pass the third parameter `handoffs_tool_overrides`. Similar to the second parameter, it is also a dictionary where the key is the agent name and the value is the corresponding handoff tool implementation.

Custom handoff tools must return a `Command` object whose `update` attribute must include the `messages` key (for returning tool responses) and the `active_agent` key (whose value is the name of the agent to hand off to, used to switch the current agent).

For example:

```python
@tool
def transfer_to_code_agent(runtime: ToolRuntime) -> Command:
    """This tool help you transfer to the code agent."""
    #You can add custom logic here
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content="transfer to code agent",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "active_agent": "code_agent",
            #You can add other keys to update here
        }
    )

agent = create_agent(
    model="vllm:qwen3-4b",
    tools=[
        get_current_time,
        get_current_weather,
        get_current_city,
        run_code,
    ],
    middleware=[
        HandoffAgentMiddleware(
            agents_config=agent_config,
            handoffs_tool_overrides={
                "code_agent": transfer_to_code_agent,
            },
        )
    ],
)
```

`handoffs_tool_overrides` is used for highly customized handoff tool implementations. If you only want to customize the description of the handoff tool, you should use `custom_handoffs_tool_descriptions`.

## Tool Call Repair

`ToolCallRepairMiddleware` is a middleware for **automatically fixing invalid tool calls (`invalid_tool_calls`) from large models**.

When large models output the JSON Schema for tool calls, they may generate JSON format errors due to the model's own reasons (errors are common in the `arguments` field), leading to JSON parsing failures. Such calls are stored in the `invalid_tool_calls` field. `ToolCallRepairMiddleware` will automatically detect `invalid_tool_calls` after the model returns results and attempt to fix them by calling `json-repair`, allowing the tool calls to execute normally.

Please ensure you have installed `langchain-dev-utils[standard]`, see the **Installation Guide** for details.

### Parameter Description

This middleware is zero-configuration and ready to use out of the box, no additional parameters required.

### Usage Example

```python
from langchain_dev_utils.agents.middleware import ToolCallRepairMiddleware

agent = create_agent(
    model="vllm:qwen3-4b",
    tools=[run_python_code, get_current_time],
    middleware=[
        ToolCallRepairMiddleware()
    ],
)
```

!!! warning "Note"
    This middleware cannot guarantee 100% fixing of all invalid tool calls; the actual effect depends on the repair capability of `json-repair`. Additionally, it only acts on invalid tool call content in the `invalid_tool_calls` field.

## Formatting System Prompts

`format_prompt` is a **middleware instance** that allows you to use `f-string` style placeholders (like `{name}`) in `system_prompt` and dynamically replace them with actual values at runtime.

### Parameter Description

The values of variables in placeholders follow a clear resolution order:

1. **First look up in `state`**: It first looks for fields with the same name as the placeholder in the `state` dictionary
2. **Then look up in `context`**: If the field is not found in `state`, it continues to look in the `context` object

This order means that values in `state` have higher priority and can override values with the same name in `context`.

### Usage Example

#### Getting Variables Only from `state`

This is the most basic usage, where all placeholder variables are provided by `state`.

```python
from langchain_dev_utils.agents.middleware import format_prompt
from langchain.agents import AgentState

class AssistantState(AgentState):
    name: str

agent = create_agent(
    model="vllm:qwen3-4b",
    system_prompt="You are an intelligent assistant, your name is {name}.",
    middleware=[format_prompt],
    state_schema=AssistantState,
)

# When calling, you must provide a value for 'name' in state
response = agent.invoke(
    {"messages": [HumanMessage(content="Hello")], "name": "assistant"}
)
print(response)
```

#### Getting Variables from Both `state` and `context`

Using both `state` and `context` simultaneously:

```python
from dataclasses import dataclass

@dataclass
class Context:
    user: str

agent = create_agent(
    model="vllm:qwen3-4b",
    # {name} will be obtained from state, {user} will be obtained from context
    system_prompt="You are an intelligent assistant, your name is {name}. Your user is named {user}.",
    middleware=[format_prompt],
    state_schema=AssistantState,
    context_schema=Context,
)

# When calling, provide 'name' for state and 'user' for context
response = agent.invoke(
    {
        "messages": [HumanMessage(content="I want to visit New York for a few days, help me plan my itinerary")],
        "name": "assistant",
    },
    context=Context(user="Zhang San"),
)
print(response)
```

#### Variable Override Example

This example shows that when `state` and `context` have variables with the same name, the value in `state` takes precedence.

```python
from dataclasses import dataclass

@dataclass
class Context:
    # 'name' is defined in context
    name: str
    user: str

agent = create_agent(
    model="vllm:qwen3-4b",
    system_prompt="You are an intelligent assistant, your name is {name}. Your user is named {user}.",
    middleware=[format_prompt],
    state_schema=AssistantState, # 'name' is also defined in state
    context_schema=Context,
)

# When calling, both state and context provide a value for 'name'
response = agent.invoke(
    {
        "messages": [HumanMessage(content="What is your name?")],
        "name": "assistant-1",
    },
    context=Context(name="assistant-2", user="Zhang San"),
)

# The final system prompt will be "You are an intelligent assistant, your name is assistant-1. Your user is named Zhang San."
# Because state has higher priority
print(response)
```

!!! warning "Note"
    There are two ways to implement custom middleware: decorators or class inheritance.  
    - Class inheritance implementation: `PlanMiddleware`, `ModelMiddleware`, `HandoffAgentMiddleware`, `ToolCallRepairMiddleware`  
    - Decorator implementation: `format_prompt` (the decorator directly turns the function into a middleware instance, so no manual instantiation is needed to use it)

!!! info "Official Middleware Extensions"
    This library extends the following official middleware, supporting model specification through string parameters for models registered with `register_model_provider`:

    You only need to import these middleware from this library to use string parameters for models registered with `register_model_provider`. The usage of the middleware remains consistent with the official middleware, for example:
    ```python
    from langchain_core.messages import AIMessage
    from langchain_dev_utils.agents.middleware import SummarizationMiddleware
    from langchain_dev_utils.chat_models import register_model_provider

    register_model_provider(
        provider_name="vllm",
        chat_model="openai-compatible",
        base_url="http://localhost:8000/v1",
    )
    agent = create_agent(
        model="vllm:qwen3-4b",
        middleware=[
            SummarizationMiddleware(
                model="vllm:qwen3-4b",
                trigger=("tokens", 50),
                keep=("messages", 1),
            )
        ],
        system_prompt="You are an intelligent AI assistant that can solve user problems",
    )
    # big_text is a text containing a lot of content, omitted here
    big_messages = [
        HumanMessage(content="Hello, who are you"),
        AIMessage(content="I am your AI assistant"),
        HumanMessage(content="Write a beautiful long text"),
        AIMessage(content=f"Okay, I will write a beautiful long text, the content is: {big_text}"),
        HumanMessage(content="Why did you write this long text?"),
    ]
    response = agent.invoke({"messages": big_messages})
    print(response)
    ```