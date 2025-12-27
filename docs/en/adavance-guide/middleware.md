# Middleware

## Overview

Middleware are components specifically built for `langchain` pre-built Agents. The official library provides some built-in middleware. This library, based on actual needs and use cases, offers additional middleware.

## Task Planning

Task planning middleware is used for structured decomposition and process management before executing complex tasks.

!!! note "Supplement"
    Task planning is an efficient context engineering management strategy. Before executing a task, the large model first breaks down the overall task into multiple ordered subtasks, forming a task planning list (called "plan" in this library). Then it executes each subtask in sequence and dynamically updates the task status after completing each step until all subtasks are finished.

The middleware that implements task planning is `PlanMiddleware`, which accepts the following parameters:

- `system_prompt`: Optional string type, system prompt. Default value is `None`, which will use the default system prompt.
- `write_plan_tool_description`: Optional string type, description of the write plan tool. Default value is `None`, which will use the default write plan tool description.
- `finish_sub_plan_tool_description`: Optional string type, description of the finish sub plan tool. Default value is `None`, which will use the default finish sub plan tool description.
- `read_plan_tool_description`: Optional string type, description of the read plan tool. Default value is `None`, which will use the default read plan tool description.
- `use_read_plan_tool`: Optional boolean type, whether to use the read plan tool. Default value is `True`, which will enable the read plan tool.


**Usage Example**:

```python
from langchain_dev_utils.agents.middleware import PlanMiddleware

agent = create_agent(
    model="vllm:qwen3-4b",
    middleware=[
        PlanMiddleware(
            use_read_plan_tool=True, # Set this parameter to False if you don't want to use the read plan tool
        )
    ],
)

response = agent.invoke(
    {"messages": [HumanMessage(content="I want to visit New York for a few days, help me plan my itinerary")]}
)
print(response)
```

`PlanMiddleware` requires the use of two tools: `write_plan` and `finish_sub_plan`, while the `read_plan` tool is enabled by default; if not needed, you can set the `use_read_plan_tool` parameter to `False`.

This middleware is similar in functionality to the **To-do list middleware** provided by LangChain officially, but there are differences in tool design. The official middleware only provides the `write_todo` tool, which is oriented towards a todo list structure; while this library provides three specialized tools: `write_plan`, `finish_sub_plan`, and `read_plan`, specifically for writing, modifying, and querying plan lists.

Whether it's `todo` or `plan`, their essence is the same, so the key difference between this middleware and the official one lies in the tools provided. The official middleware completes addition and modification through one tool, while this library provides three tools, where `write_plan` can be used to write plans or update plan content, `finish_sub_plan` is used to update the status after completing a subtask, and `read_plan` is used to query plan content.

At the same time, this library also provides three functions to create these three tools:

- `create_write_plan_tool`: A function to create a tool for writing plans
- `create_finish_sub_plan_tool`: A function to create a tool for completing subtasks
- `create_read_plan_tool`: A function to create a tool for querying plans

These three functions can all receive a `description` parameter for customizing the tool description. If not passed in, the default tool description will be used. Among them, `create_write_plan_tool` and `create_finish_sub_plan_tool` can also receive a `message_key` parameter for customizing the key to update messages. If not passed in, the default `messages` key will be used.

**Usage Example**:

```python
from langchain_dev_utils.agents.middleware.plan import (
    create_write_plan_tool,
    create_finish_sub_plan_tool,
    create_read_plan_tool,
    PlanState,
)

agent = create_agent(
    model="vllm:qwen3-4b",
    state_schema=PlanState,
    tools=[create_write_plan_tool(), create_finish_sub_plan_tool(), create_read_plan_tool()],
)
```

It should be noted that to use these three tools, you must ensure that the state Schema contains the plan key, otherwise an error will be reported. For this, you can use the `PlanState` provided by this library to inherit the state Schema.

!!! success "Best Practice"
    1. When using `create_agent`:

    It is recommended to directly use `PlanMiddleware` instead of manually passing in the three tools `write_plan`, `finish_sub_plan`, and `read_plan`.

    Reason: The middleware has automatically handled prompt construction and agent state management, which can significantly reduce usage complexity.

    Note: Since the model output of `create_agent` is fixed to update to the `messages` key, `PlanMiddleware` does not have a `message_key` parameter.

    2. When using `langgraph`:

    It is recommended to directly use these three tools (`write_plan`, `finish_sub_plan`, `read_plan`).

    Reason: This approach can better integrate with `langgraph`'s custom nodes and state management.


## Model Routing

`ModelRouterMiddleware` is a middleware used for **dynamically routing to the most suitable model based on input content**. It analyzes user requests through a "routing model" and selects the most appropriate model from a predefined list of models to handle the current task.

Its parameters are as follows:

- `router_model`: The model used to execute routing decisions. You can pass in a string (which will be automatically loaded through `load_chat_model`), such as `vllm:qwen3-4b`; or directly pass in an instantiated `BaseChatModel` object.
- `model_list`: List of model configurations, where each element is a dictionary that can contain the following fields:
    - `model_name` (str): Required, the unique identifier of the model, **using `provider:model-name` format**, such as `vllm:qwen3-4b`;
    - `model_description` (str): Required, a brief description of the model's capabilities or applicable scenarios, for the routing model to make decisions.
    - `tools` (list[BaseTool]): Optional, the whitelist of tools that this model can call. If not provided, it inherits the global tool list; if set to `[]`, it explicitly disables all tools.
    - `model_kwargs` (dict): Optional, additional parameters when loading the model (such as `temperature`, `max_tokens`, etc.).
    - `model_system_prompt` (str): Optional, the system-level prompt of the model.
    - `model_instance` (BaseChatModel): Optional, an instantiated model object.


!!! tip "Note"
    The following is an explanation for the `model_instance` field:
    
    - If provided, this instance is used directly, `model_name` is only for identification, and `model_kwargs` is ignored; suitable for situations where the conversation model management function of this library is not used.

    - If not provided, the model will be loaded using `load_chat_model` according to `model_name` and `model_kwargs`.

    - In either case, it is recommended to name `model_name` in the `provider:model-name` format.


- `router_prompt`: Custom prompt for the routing model. If `None` (default), the built-in default prompt template is used.


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
        "model_description": "Suitable for visual tasks",
        "tools": [],  # If this model doesn't need any tools, please set this field to an empty list []
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
        ),  # Directly pass in the instance, at this time model_name is only for identification, model_kwargs is ignored
    },
]
```


Then enable the middleware when creating the agent:

```python
from langchain_dev_utils.agents.middleware import ModelRouterMiddleware
from langchain_core.messages import HumanMessage

agent = create_agent(
    model="vllm:qwen3-4b",  # This model is only a placeholder, actually dynamically replaced by the middleware
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

Through `ModelRouterMiddleware`, you can easily build a multi-model, multi-capability Agent that automatically selects the optimal model based on task type, improving response quality and efficiency.


## Handoffs Middleware
`HandoffsAgentMiddleware` is a middleware used for **flexibly switching between multiple sub-Agents**, fully implementing LangChain's official `handoffs` multi-agent collaboration solution.

Its parameters are as follows:

- `agents_config`: Dictionary of agent configurations, where keys are agent names and values are agent configuration dictionaries.
    - `model` (str | BaseChatModel): Optional, specifies the model used by this agent; if not passed, it will use the model corresponding to the `model` parameter of `create_agent`. Supports strings (must be in `provider:model-name` format, such as `vllm:qwen3-4b`) or `BaseChatModel` instances.  
    - `prompt` (str | SystemMessage): Required, the system prompt of the agent.  
    - `tools` (list[BaseTool]): Required, the list of tools that the agent can call.  
    - `default` (bool): Optional, whether to set as the default agent; default is `False`. There must be and can only be one agent set to `True` in all configurations.


For this paradigm of multi-agent implementation, a tool for handoffs is often needed. This library provides a tool method `create_handoffs_tool` for creating handoff tools. This tool function receives three parameters, as follows:

- `agent_name`: Required, represents the name of the target agent;
- `tool_name`: Optional, represents the tool name, default value is `transfer_to_{agent_name}`;
- `tool_description`: Optional, represents the tool description, if not passed, the default tool description is used.


**Usage Example**

In this example, we will use four agents: `time_agent`, `weather_agent`, `code_agent`, and `default_agent`.

First, we need to create the corresponding four handoff tools:

```python
from langchain_dev_utils.agents.middleware import create_handoffs_tool

transfer_to_time_agent = create_handoffs_tool("time_agent")
transfer_to_weather_agent = create_handoffs_tool("weather_agent")
transfer_to_code_agent = create_handoffs_tool("code_agent")
transfer_to_default_agent = create_handoffs_tool("default_agent")
```

Next, we need to create the corresponding agent configuration dictionary `agent_config`. It should be noted that the name of each agent (i.e., the key of the dictionary) must be consistent with the `agent_name` parameter of the created handoff tool.

```python
from langchain_dev_utils.agents.middleware.handoffs import AgentConfig

agent_config: dict[str, AgentConfig] = {
    "time_agent": {
        "model": "vllm:qwen3-8b",
        "prompt": "You are a time assistant",
        "tools": [
            get_current_time,
            transfer_to_weather_agent,
            transfer_to_code_agent,
            transfer_to_default_agent,
        ],
    },
    "weather_agent": {
        "prompt": "You are a weather assistant",
        "tools": [
            get_current_weather,
            get_current_city,
            transfer_to_default_agent,
            transfer_to_time_agent,
            transfer_to_code_agent,
        ],
    },
    "code_agent": {
        "model": load_chat_model("vllm:qwen3-coder-flash"),
        "prompt": "You are a code assistant",
        "tools": [
            run_code,
            transfer_to_default_agent,
            transfer_to_time_agent,
            transfer_to_weather_agent,
        ],
    },
    "default_agent": {
        "prompt": "You are an assistant",
        "tools": [
            transfer_to_time_agent,
            transfer_to_weather_agent,
            transfer_to_code_agent,
        ],
        "default": True,
    },
}
```

Finally, pass this configuration to `HandoffsAgentMiddleware`.

```python
from langchain_dev_utils.agents.middleware import HandoffsAgentMiddleware

agent = create_agent(
    model="vllm:qwen3-4b",
    tools=[
        transfer_to_time_agent,
        transfer_to_weather_agent,
        transfer_to_code_agent,
        transfer_to_default_agent,
        get_current_time,
        get_current_weather,
        get_current_city,
        run_code,
    ],
    middleware=[HandoffsAgentMiddleware(agents_config=agent_config)],
)

response = agent.invoke({"messages":[HumanMessage(content="What is the current time?")]})
print(response)
```

## Tool Call Repair
`ToolCallRepairMiddleware` is a middleware that **automatically repairs invalid tool calls (`invalid_tool_calls`) by large models**.

When large models output JSON Schema for tool calls, they may generate JSON format error content due to the model itself (error content is common in the `arguments` field), causing JSON parsing to fail. These calls will be stored in the `invalid_tool_calls` field. `ToolCallRepairMiddleware` will automatically detect `invalid_tool_calls` after the model returns results and try to call `json-repair` for repair, so that tool calls can be executed normally.

Please ensure that `langchain-dev-utils[standard]` is installed, see the **Installation Guide** for details.

This middleware is zero-configuration and ready to use out of the box, no additional parameters are required.

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

!!! warning "Note"
    This middleware cannot guarantee 100% repair of all invalid tool calls; the actual effect depends on the repair capability of `json-repair`; additionally, it only acts on invalid tool call content in the `invalid_tool_calls` field.


## Formatting System Prompts

This middleware `format_prompt` allows you to use `f-string` style placeholders (such as `{name}`) in `system_prompt`, and dynamically replace them with actual values at runtime.

The values of variables in placeholders follow a clear parsing order:

1.  **First look up from `state`**: It will first look up fields with the same name as the placeholder in the `state` dictionary.
2.  **Then look up from `context`**: If the field is not found in `state`, it will continue to look up in the `context` object.

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

    Using both `state` and `context`:

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

    This example shows that when there are variables with the same name in `state` and `context`, the value of `state` will take effect first.

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

    # When calling, both state and context provide values for 'name'
    response = agent.invoke(
        {
            "messages": [HumanMessage(content="What is your name?")],
            "name": "assistant-1",
        },
        context=Context(name="assistant-2", user="Zhang San"),
    )

    # The final system prompt will be "You are an intelligent assistant, your name is assistant-1. Your user is called Zhang San."
    # Because the priority of state is higher
    print(response)
    ```

!!! warning "Note"
    There are two ways to implement custom middleware: decorator or inheritance class.  
    - Inheritance class implementation: `PlanMiddleware`, `ModelMiddleware`, `HandoffsAgentMiddleware`, `ToolCallRepairMiddleware`  
    - Decorator implementation: `format_prompt` (the decorator directly turns the function into a middleware instance, so no manual instantiation is required to use it)


!!! info "Note"
    In addition, this library has also expanded the following middleware to support the function of specifying models through string parameters:

    - SummarizationMiddleware

    - LLMToolSelectorMiddleware

    - ModelFallbackMiddleware
    
    - LLMToolEmulator

    You just need to import these middleware from this library to use strings to specify models that have been registered by `register_model_provider`. The usage of the middleware is consistent with the official middleware, for example:
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