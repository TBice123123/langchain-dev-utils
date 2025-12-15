# Middleware

## Overview

Middleware are components specifically built for LangChain's pre-built Agents. The official library provides some built-in middleware. This library offers additional middleware based on practical needs and usage scenarios.

## Task Planning

Task planning middleware is used for structured decomposition and process management before executing complex tasks.

!!! note "Additional Information"
    Task planning is an efficient context engineering management strategy. Before executing a task, the large language model first breaks down the overall task into multiple ordered subtasks, forming a task planning list (called a "plan" in this library). Then it executes each subtask in sequence and dynamically updates the task status after completing each step until all subtasks are finished.

The middleware that implements task planning is `PlanMiddleware`, which accepts the following parameters:

- `system_prompt`: Optional string type, system prompt. Default value is `None`, which will use the default system prompt.
- `write_plan_tool_description`: Optional string type, description of the write plan tool. Default value is `None`, which will use the default write plan tool description.
- `finish_sub_plan_tool_description`: Optional string type, description of the finish sub-plan tool. Default value is `None`, which will use the default finish sub-plan tool description.
- `read_plan_tool_description`: Optional string type, description of the read plan tool. Default value is `None`, which will use the default read plan tool description.
- `use_read_plan_tool`: Optional boolean type, whether to use the read plan tool. Default value is `True`, which will enable the read plan tool.

**Usage Example**:

```python
from langchain_dev_utils.agents.middleware import PlanMiddleware

agent = create_agent(
    model="vllm:qwen3-4b",
    middleware=[
        PlanMiddleware(
            use_read_plan_tool=True, # If you don't want to use the read plan tool, you can set this parameter to False
        )
    ],
)

response = agent.invoke(
    {"messages": [HumanMessage(content="I want to visit New York for a few days, help me plan my itinerary")]}
)
print(response)
```

`PlanMiddleware` requires the use of `write_plan` and `finish_sub_plan` tools, while the `read_plan` tool is enabled by default; if not needed, you can set the `use_read_plan_tool` parameter to `False`.

This middleware is similar in functionality to the **To-do list middleware** provided by LangChain officially, but there are differences in tool design. The official middleware only provides a `write_todo` tool, targeting a todo list structure; while this library provides three specialized tools: `write_plan`, `finish_sub_plan`, and `read_plan`, specifically for writing, modifying, and querying plan lists.

Whether it's "todo" or "plan", they are essentially the same thing. Therefore, the key difference between this middleware and the official one lies in the tools provided. The official middleware uses one tool for adding and modifying, while this library provides three tools: `write_plan` can be used to write plans or update plan content, `finish_sub_plan` is used to update the status after completing a subtask, and `read_plan` is used to query plan content.

At the same time, this library also provides three functions to create these three tools:

- `create_write_plan_tool`: A function to create a tool for writing plans
- `create_finish_sub_plan_tool`: A function to create a tool for completing subtasks
- `create_read_plan_tool`: A function to create a tool for querying plans

All three functions can receive a `description` parameter to customize the tool's description. If not provided, the default tool description will be used. Additionally, `create_write_plan_tool` and `create_finish_sub_plan_tool` can receive a `message_key` parameter to customize the key for updating messages. If not provided, the default `messages` key will be used.

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

Note that to use these three tools, you must ensure that the state Schema contains the "plan" key, otherwise it will raise an error. For this, you can use the `PlanState` provided by this library to inherit the state Schema.

!!! note "Best Practices"
    1. When using `create_agent`:

    It is recommended to directly use `PlanMiddleware` rather than manually passing in the three tools: `write_plan`, `finish_sub_plan`, and `read_plan`.

    Reason: The middleware has already handled prompt construction and agent state management, which can significantly reduce usage complexity.

    Note: Since the model output of `create_agent` is fixed to update to the `messages` key, `PlanMiddleware` does not have a `message_key` parameter.

    2. When using `langgraph`:

    It is recommended to directly use these three tools (`write_plan`, `finish_sub_plan`, `read_plan`).

    Reason: This approach can better integrate with the custom nodes and state management of `langgraph`.

## Model Routing

`ModelRouterMiddleware` is a middleware used for **dynamically routing to the most suitable model based on input content**. It analyzes user requests through a "routing model" and selects the most suitable model from a predefined list for processing.

Its parameters are as follows:

- `router_model`: The model used to execute routing decisions. You can pass a string (which will be automatically loaded via `load_chat_model`), such as `vllm:qwen3-4b`; or directly pass an instantiated `BaseChatModel` object.
- `model_list`: A list of model configurations, where each element is a dictionary that can contain the following fields:
    - `model_name` (str): Required, the unique identifier of the model, **using the `provider:model-name` format**, such as `vllm:qwen3-4b` or `openrouter:qwen/qwen3-vl-32b-instruct`;
    - `model_description` (str): Required, a brief description of the model's capabilities or applicable scenarios for the routing model to make decisions.
    - `tools` (list[BaseTool]): Optional, the whitelist of tools that the model can call.
            - If not provided, it inherits the global tool list;
            - If set to `[]`, it explicitly disables all tools.
    - `model_kwargs` (dict): Optional, additional parameters when loading the model (such as `temperature`, `max_tokens`, etc.), **only effective when `model_instance` is not passed**.
    - `model_instance` (BaseChatModel): Optional, an instantiated model object.
            - If provided, this instance is used directly, `model_name` is only for identification, **no longer loaded via `load_chat_model`**, and `model_kwargs` is ignored;
            - If not provided, the model is automatically loaded according to `model_name` and `model_kwargs`.
    - `model_system_prompt` (str): Optional, the system-level prompt of the model.
- `router_prompt`: Custom prompt for the routing model. If `None` (default), the built-in default prompt template is used.

**Usage Example**

First define the model list:

```python
model_list = [
    {
        "model_name": "vllm:qwen3-8b",
        "model_description": "Suitable for ordinary tasks, such as conversations, text generation, etc.",
        "model_kwargs": {
            "temperature": 0.7,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
        },
        "model_system_prompt": "You are an assistant, good at handling ordinary tasks, such as conversations, text generation, etc.",
    },
    {
        "model_name": "openrouter:qwen/qwen3-vl-32b-instruct",
        "model_description": "Suitable for vision tasks",
        "tools": [],  # If this model doesn't need any tools, please set this field to an empty list []
    },
    {
        "model_name": "openrouter:qwen/qwen3-coder-plus",
        "model_description": "Suitable for code generation tasks",
        "tools": [run_python_code],  # Only allow the use of run_python_code tool
    },
]
```

Then enable the middleware when creating the agent:

```python
from langchain_dev_utils.agents.middleware import ModelRouterMiddleware
from langchain_core.messages import HumanMessage

agent = create_agent(
    model="vllm:qwen3-4b",  # This model is only a placeholder, actually replaced dynamically by the middleware
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

## Tool Call Repair
`ToolCallRepairMiddleware` is a middleware that **automatically repairs invalid tool calls (`invalid_tool_calls`) by large language models**.

When large language models output JSON Schema for tool calls, they may generate JSON format errors due to the model's own reasons (errors are common in the `arguments` field), causing JSON parsing to fail. These calls will be stored in the `invalid_tool_calls` field. `ToolCallRepairMiddleware` will automatically detect `invalid_tool_calls` after the model returns results and attempt to repair them by calling `json-repair`, so that tool calls can be executed normally.

Please ensure that `langchain-dev-utils[standard]` is installed, see the **Installation Guide** for details.

This middleware is zero-configuration and ready to use out of the box, with no additional parameters required.

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
    This middleware cannot guarantee to repair all invalid tool calls 100%. The actual effect depends on the repair capability of `json-repair`; additionally, it only acts on invalid tool call content in the `invalid_tool_calls` field.

## Format System Prompt

This middleware `format_prompt` allows you to use `f-string` style placeholders (such as `{name}`) in `system_prompt` and dynamically replace them with actual values at runtime.

The values of variables in placeholders follow a clear parsing order:

1.  **Priority lookup from `state`**: First look for fields with the same name as the placeholder in the `state` dictionary.
2.  **Then lookup from `context`**: If the field is not found in `state`, continue searching in the `context` object.

This order means that values in `state` have higher priority and can override values with the same name in `context`.

Usage examples are as follows:

- **Get variables only from `state`**

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

- **Get variables from both `state` and `context`**

    Use both `state` and `context` simultaneously:

    ```python
    from dataclasses import dataclass

    @dataclass
    class Context:
        user: str

    agent = create_agent(
        model="vllm:qwen3-4b",
        # {name} will be obtained from state, {user} will be obtained from context
        system_prompt="You are an intelligent assistant, your name is {name}. Your user's name is {user}.",
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

    This example shows that when there are variables with the same name in `state` and `context`, the value of `state` takes precedence.

    ```python
    from dataclasses import dataclass

    @dataclass
    class Context:
        # 'name' is defined in context
        name: str
        user: str

    agent = create_agent(
        model="vllm:qwen3-4b",
        system_prompt="You are an intelligent assistant, your name is {name}. Your user's name is {user}.",
        middleware=[format_prompt],
        state_schema=AssistantState, # 'name' is also defined in state
        context_schema=Context,
    )

    # When calling, both state and context provide values for 'name'
    response = agent.invoke(
        {
            "messages": [HumanMessage(content="What's your name?")],
            "name": "assistant-1",
        },
        context=Context(name="assistant-2", user="Zhang San"),
    )

    # The final system prompt will be "You are an intelligent assistant, your name is assistant-1. Your user's name is Zhang San."
    # Because state has higher priority
    print(response)
    ```

!!! warning "Note"
    There are two ways to implement custom middleware: decorator or class inheritance.
    - Class inheritance implementation: `PlanMiddleware`, `ModelMiddleware`, `ToolCallRepairMiddleware`
    - Decorator implementation: `format_prompt` (the decorator directly converts the function into a middleware instance, so no manual instantiation is required)

!!! info "Note"
    In addition, this library has also expanded the following middleware to support the functionality of specifying models through string parameters:

    - SummarizationMiddleware

    - LLMToolSelectorMiddleware

    - ModelFallbackMiddleware
    
    - LLMToolEmulator

    You only need to import these middleware from this library to use strings to specify models that have been registered with `register_model_provider`. The usage of the middleware is consistent with the official middleware, for example:
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