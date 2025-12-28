# Middleware


## Overview

Middleware are components specifically built for pre-built Agents in `langchain`. The official library provides some built-in middleware. This library, based on practical needs and use cases, offers additional middleware.

## Task Planning

Task planning middleware is used for structured decomposition and process management before executing complex tasks.

!!! note "Note"
    Task planning is an efficient context engineering management strategy. Before executing a task, the large model first breaks down the overall task into multiple ordered subtasks, forming a task planning list (called a plan in this library). It then executes each subtask in sequence and dynamically updates the task status after completing each step until all subtasks are finished.

The middleware that implements task planning is `PlanMiddleware`, which accepts the following parameters:

- `system_prompt`: Optional string type, system prompt. Default value is `None`, which will use the default system prompt.
- `custom_plan_tool_descriptions`: Optional dictionary for customizing the descriptions of plan-related tools. Its keys can take the following three values:
    - `write_plan`: Description of the write plan tool  
    - `finish_sub_plan`: Description of the finish subplan tool  
    - `read_plan`: Description of the read plan tool  
- `use_read_plan_tool`: Optional boolean type, whether to use the read plan tool. Default value is `True`, which will enable the read plan tool.


**Usage Example**:

```python
from langchain_dev_utils.agents.middleware import PlanMiddleware

agent = create_agent(
    model="vllm:qwen3-4b",
    middleware=[
        PlanMiddleware(
            custom_plan_tool_descriptions={
                "write_plan": "Used to write a plan, breaking down the task into multiple ordered subtasks.",
                "finish_sub_plan": "Used to complete a subplan, updating the subtask status to completed.",
                "read_plan": "Used to query the current task planning list."
            },
            use_read_plan_tool=True, # If you don't want to use the read plan tool, set this parameter to False
        )
    ],
)

response = agent.invoke(
    {"messages": [HumanMessage(content="I want to visit New York for a few days, help me plan my itinerary")]}
)
print(response)
```

`PlanMiddleware` requires the use of `write_plan` and `finish_sub_plan` tools, while the `read_plan` tool is enabled by default; if not needed, the `use_read_plan_tool` parameter can be set to `False`.

This middleware is similar in functionality to the **To-do list middleware** provided by LangChain officially, but there are differences in tool design. The official middleware only provides the `write_todo` tool, targeting the todo list structure; while this library provides three specialized tools: `write_plan`, `finish_sub_plan`, and `read_plan`, specifically for writing, modifying, and querying plan lists.

Whether it's `todo` or `plan`, their essence is the same, so the key difference between this middleware and the official one lies in the tools provided. The official one uses one tool for both adding and modifying, while this library provides three tools: `write_plan` can be used to write or update plan content, `finish_sub_plan` is used to update the status after completing a subtask, and `read_plan` is used to query plan content.

## Model Routing

`ModelRouterMiddleware` is a middleware used to **dynamically route to the most suitable model based on input content**. It analyzes user requests through a "routing model" and selects the most suitable model from a predefined list for processing the current task.

Its parameters are as follows:

- `router_model`: The model used to execute routing decisions. Can be passed as a string (will be automatically loaded via `load_chat_model`), such as `vllm:qwen3-4b`; or directly pass an instantiated `BaseChatModel` object.
- `model_list`: List of model configurations, each element is a dictionary that can contain the following fields:
    - `model_name` (str): Required, unique identifier of the model, **using `provider:model-name` format**, such as `vllm:qwen3-4b`;
    - `model_description` (str): Required, brief description of the model's capabilities or applicable scenarios for the routing model to make decisions.
    - `tools` (list[BaseTool]): Optional, whitelist of tools that this model can call. If not provided, it inherits the global tool list; if set to `[]`, it explicitly disables all tools.
    - `model_kwargs` (dict): Optional, additional parameters when loading the model (such as `temperature`, `max_tokens`, etc.).
    - `model_system_prompt` (str): Optional, system-level prompt of the model.
    - `model_instance` (BaseChatModel): Optional, already instantiated model object.


!!! tip "Note"
    Regarding the `model_instance` field:
    
    - If provided, the instance is used directly, `model_name` is only for identification, and `model_kwargs` is ignored; suitable for cases not using the conversation model management function of this library.

    - If not provided, the model will be loaded using `load_chat_model` according to `model_name` and `model_kwargs`.

    - In either case, it's recommended to use the `provider:model-name` format for `model_name` naming.


- `router_prompt`: Custom prompt for the routing model. If `None` (default), the built-in default prompt template will be used.


**Usage Example**

First define the model list:

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
        "model_description": "Suitable for vision tasks",
        "tools": [],  # If this model doesn't need any tools, set this field to an empty list []
    },
    {
        "model_name": "vllm:qwen3-coder-flash",
        "model_description": "Suitable for code generation tasks",
        "tools": [run_python_code],  # Only allow the use of run_python_code tool
    },
    {
        "model_name": "openai:gpt-4o",
        "model_description": "Suitable for comprehensive and difficult tasks",
        "model_system_prompt": "You are an assistant, good at handling comprehensive and difficult tasks",
        "model_instance": ChatOpenAI(
            model_name="gpt-4o"
        ),  # Directly pass the instance, at this time model_name is only for identification, model_kwargs is ignored
    },
]
```


Then enable the middleware when creating the agent:

```python
from langchain_dev_utils.agents.middleware import ModelRouterMiddleware
from langchain_core.messages import HumanMessage

agent = create_agent(
    model="vllm:qwen3-4b",  # This model is just a placeholder, actually replaced dynamically by the middleware
    tools=[run_python_code, get_current_time],
    middleware=[
        ModelRouterMiddleware(
            router_model="vllm:qwen3-4b",
            model_list=model_list,
        )
    ],
)

# The routing middleware will automatically select the most suitable model based on the input content
response = agent.invoke({"messages": [HumanMessage(content="Help me write a bubble sort code")]})
print(response)
```

Through `ModelRouterMiddleware`, you can easily build a multi-model, multi-capability Agent that automatically selects the optimal model according to the task type, improving response quality and efficiency.


## Handoffs Middleware
`HandoffAgentMiddleware` is a middleware used to **flexibly switch between multiple sub-Agents**, fully implementing LangChain's official `handoffs` multi-agent collaboration solution.

Its parameters are as follows:

- `agents_config`: Dictionary of agent configurations, with agent names as keys and agent configuration dictionaries as values.
    - `model` (str | BaseChatModel): Optional, specifies the model used by this agent; if not passed, it follows the model corresponding to the `model` parameter of `create_agent`. Supports strings (must be in `provider:model-name` format, such as `vllm:qwen3-4b`) or `BaseChatModel` instances.  
    - `prompt` (str | SystemMessage): Required, system prompt of the agent.  
    - `tools` (list[BaseTool]): Optional, list of tools that the agent can call.  
    - `default` (bool): Optional, whether to set as the default agent; default is `False`. There must be and can only be one agent set to `True` in all configurations.
    - `handoffs` (list[str] | str): Required, list of other agent names that this agent can hand off to. If set to `"all"`, it means this agent can hand off to all other agents.
- `custom_handoffs_tool_descriptions` (dict[str, str]): Optional, custom descriptions of handoff tools. Keys are agent names, values are descriptions of the corresponding handoff tools. If not provided (i.e., default value `None`), the built-in default descriptions are used.


For this paradigm of multi-agent implementation, a tool for handoffs is often needed. This middleware utilizes the `handoffs` configuration of each agent to automatically create corresponding handoff tools for each agent. If you want to customize the description of handoff tools, you can achieve this through the `custom_handoffs_tool_descriptions` parameter.


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
        "prompt": "You are a code assistant",
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

Finally, just pass this configuration to `HandoffAgentMiddleware`.

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

If you want to customize the description of handoff tools, you can pass the second parameter `custom_handoffs_tool_descriptions`.

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
                "time_agent": "This tool is used to hand off to the time assistant to solve time query problems",
                "weather_agent": "This tool is used to hand off to the weather assistant to solve weather query problems",
                "code_agent": "This tool is used to hand off to the code assistant to solve code problems",
                "default_agent": "This tool is used to hand off to the default assistant",
            },
        )
    ],
)
```

## Tool Call Repair
`ToolCallRepairMiddleware` is a middleware that **automatically repairs invalid tool calls (`invalid_tool_calls`) from large models**.

When large models output JSON Schema for tool calls, they may generate JSON format errors due to the model itself (common errors in the `arguments` field), causing JSON parsing to fail. These calls will be stored in the `invalid_tool_calls` field. `ToolCallRepairMiddleware` will automatically detect `invalid_tool_calls` after the model returns results and attempt to use `json-repair` to fix them, allowing tool calls to execute normally.

Make sure you have installed `langchain-dev-utils[standard]`, see the **Installation Guide** for details.

This middleware is zero-configuration and ready to use out of the box, with no additional parameters needed.

**Usage Example:**

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

!!! warning "Warning"
    This middleware cannot guarantee 100% repair of all invalid tool calls; the actual effect depends on the repair capability of `json-repair`; additionally, it only acts on invalid tool call content in the `invalid_tool_calls` field.


## Format System Prompt

This middleware `format_prompt` allows you to use `f-string` style placeholders (such as `{name}`) in `system_prompt` and dynamically replace them with actual values at runtime.

The values of variables in placeholders follow a clear parsing order:

1.  **Priority search from `state`**: It will first search for fields with the same name as the placeholder in the `state` dictionary.
2.  **Then search from `context`**: If the field is not found in `state`, it will continue to search in the `context` object.

This order means that values in `state` have higher priority and can override values with the same name in `context`.

Usage examples are as follows:

- **Getting variables only from `state`**

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

    # When calling, you must provide the value of 'name' for state
    response = agent.invoke(
        {"messages": [HumanMessage(content="Hello")], "name": "assistant"}
    )
    print(response)
    ```

- **Getting variables from both `state` and `context`**

    Using both `state` and `context` simultaneously:

    ```python
    from dataclasses import dataclass

    @dataclass
    class Context:
        user: str

    agent = create_agent(
        model="vllm:qwen3-4b",
        # {name} will be obtained from state, {user} will be obtained from context
        system_prompt="You are an intelligent assistant, your name is {name}. Your user is called {user}.",
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

- **Variable Override Example**

    This example shows that when there are variables with the same name in `state` and `context`, the value of `state` takes effect first.

    ```python
    from dataclasses import dataclass

    @dataclass
    class Context:
        # 'name' is defined in context
        name: str
        user: str

    agent = create_agent(
        model="vllm:qwen3-4b",
        system_prompt="You are an intelligent assistant, your name is {name}. Your user is called {user}.",
        middleware=[format_prompt],
        state_schema=AssistantState, # 'name' is also defined in state
        context_schema=Context,
    )

    # When calling, both state and context provide the value of 'name'
    response = agent.invoke(
        {
            "messages": [HumanMessage(content="What's your name?")],
            "name": "assistant-1",
        },
        context=Context(name="assistant-2", user="Zhang San"),
    )

    # The final system prompt will be "You are an intelligent assistant, your name is assistant-1. Your user is called Zhang San."
    # Because state has higher priority
    print(response)
    ```

!!! warning "Warning"
    There are two ways to implement custom middleware: decorators or class inheritance.  
    - Class inheritance implementation: `PlanMiddleware`, `ModelMiddleware`, `HandoffAgentMiddleware`, `ToolCallRepairMiddleware`  
    - Decorator implementation: `format_prompt` (the decorator directly turns the function into a middleware instance, so it can be used without manual instantiation)


!!! info "Note"
    In addition, this library has expanded the following middleware to support the function of specifying models through string parameters:

    - SummarizationMiddleware

    - LLMToolSelectorMiddleware

    - ModelFallbackMiddleware
    
    - LLMToolEmulator

    You just need to import these middleware from this library, and you can use strings to specify models that have been registered with `register_model_provider`. The usage of middleware is consistent with the official middleware, for example:
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