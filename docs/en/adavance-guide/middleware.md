# Middleware

## Overview

Middleware is specifically built for pre-built `langchain` Agents. The official library provides some built-in middleware. This library, based on practical needs and use cases, offers additional middleware.

## Task Planning

The task planning middleware is used for structured decomposition and process management before executing complex tasks.

!!! note "Additional Information"
    Task planning is an efficient context engineering management strategy. Before executing a task, the large language model first breaks down the overall task into multiple ordered subtasks, forming a task planning list (referred to as "plan" in this library). Then, it executes each subtask in sequence and dynamically updates the task status after completing each step until all subtasks are finished.

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

This middleware is similar in function to the **To-do list middleware** officially provided by LangChain, but there are differences in tool design. The official middleware only provides the `write_todo` tool, which is oriented towards a todo list structure; while this library provides three specialized tools: `write_plan`, `finish_sub_plan`, and `read_plan`, specifically for writing, modifying, and querying plan lists.

Whether it's a "todo" or a "plan", their essence is the same. Therefore, the key difference between this middleware and the official one lies in the tools provided. The official middleware uses one tool for both adding and modifying, while this library provides three tools: `write_plan` for writing or updating plan content, `finish_sub_plan` for updating the status after completing a subtask, and `read_plan` for querying plan content.

Additionally, this library provides three functions to create these three tools:

- `create_write_plan_tool`: Function to create a tool for writing plans
- `create_finish_sub_plan_tool`: Function to create a tool for completing subtasks
- `create_read_plan_tool`: Function to create a tool for querying plans

These three functions can all accept a `description` parameter for customizing the tool's description. If not provided, the default tool description will be used. `create_write_plan_tool` and `create_finish_sub_plan_tool` can also accept a `message_key` parameter for customizing the key for updating messages. If not provided, the default `messages` key will be used.

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

It's important to note that to use these three tools, you must ensure that the state Schema contains the "plan" key, otherwise, an error will occur. For this, you can use the `PlanState` provided by this library to inherit the state Schema.

!!! success "Best Practices"
    1. When using `create_agent`:
    
       It's recommended to directly use `PlanMiddleware` instead of manually passing in the three tools `write_plan`, `finish_sub_plan`, and `read_plan`.
       
       Reason: The middleware automatically handles prompt construction and agent state management, significantly reducing usage complexity.
       
       Note: Since the model output of `create_agent` is fixed to update to the `messages` key, `PlanMiddleware` does not have a `message_key` parameter.
    
    2. When using `langgraph`:
    
       It's recommended to directly use these three tools (`write_plan`, `finish_sub_plan`, `read_plan`).
       
       Reason: This approach better integrates with `langgraph`'s custom nodes and state management.

## Model Routing

`ModelRouterMiddleware` is a middleware for **dynamically routing to the most suitable model based on input content**. It analyzes user requests through a "routing model" and selects the most suitable model from a predefined list for processing the current task.

Its parameters are as follows:

- `router_model`: The model used to execute routing decisions. Can be passed as a string (which will be automatically loaded via `load_chat_model`), such as `vllm:qwen3-4b`; or directly as an instantiated `BaseChatModel` object.
- `model_list`: A list of model configurations, where each element is a dictionary that can contain the following fields:
    - `model_name` (str): Required, unique identifier for the model, **using `provider:model-name` format**, for example `vllm:qwen3-4b`;
    - `model_description` (str): Required, a brief description of the model's capabilities or suitable scenarios for the routing model to make decisions.
    - `tools` (list[BaseTool]): Optional, a whitelist of tools that this model can call. If not provided, it inherits the global tool list; if set to `[]`, all tools are explicitly disabled.
    - `model_kwargs` (dict): Optional, additional parameters for model loading (such as `temperature`, `max_tokens`, etc.).
    - `model_system_prompt` (str): Optional, system-level prompt for the model.
    - `model_instance` (BaseChatModel): Optional, an already instantiated model object.

!!! tip "Note"
    Regarding the `model_instance` field:
    
    - If provided, the instance is used directly, `model_name` is only for identification, and `model_kwargs` are ignored; suitable for cases where the library's conversation model management functionality is not used.

    - If not provided, the model is loaded using `load_chat_model` based on `model_name` and `model_kwargs`.

    - In either case, it's recommended to use the `provider:model-name` format for `model_name` naming.

- `router_prompt`: Custom prompt for the routing model. If `None` (default), the built-in default prompt template is used.

**Usage Example**

First, define the model list:

```python
model_list = [
    {
        "model_name": "vllm:qwen3-8b",
        "model_description": "Suitable for general tasks, such as conversation, text generation, etc.",
        "model_kwargs": {
            "temperature": 0.7,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
        },
        "model_system_prompt": "You are an assistant, skilled at handling general tasks, such as conversation, text generation, etc.",
    },
    {
        "model_name": "vllm:qwen3-vl-2b",
        "model_description": "Suitable for vision tasks",
        "tools": [],  # If this model doesn't need any tools, set this field to an empty list []
    },
    {
        "model_name": "vllm:qwen3-coder-flash",
        "model_description": "Suitable for code generation tasks",
        "tools": [run_python_code],  # Only allow using the run_python_code tool
    },
    {
        "model_name": "openai:gpt-4o",
        "model_description": "Suitable for comprehensive and high-difficulty tasks",
        "model_system_prompt": "You are an assistant, skilled at handling comprehensive and high-difficulty tasks",
        "model_instance": ChatOpenAI(model_name="gpt-4o"), # Directly pass the instance, where model_name is only for identification and model_kwargs are ignored
    }
]
```

Then enable the middleware when creating the agent:

```python
from langchain_dev_utils.agents.middleware import ModelRouterMiddleware
from langchain_core.messages import HumanMessage

agent = create_agent(
    model="vllm:qwen3-4b",  # This model is just a placeholder, actually dynamically replaced by the middleware
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

With `ModelRouterMiddleware`, you can easily build a multi-model, multi-capability Agent that automatically selects the optimal model based on task type, improving response quality and efficiency.

## Tool Call Repair

`ToolCallRepairMiddleware` is a middleware that **automatically repairs invalid tool calls (`invalid_tool_calls`) by the large language model**.

When the large language model outputs JSON Schema for tool calls, it may generate JSON format errors in the content due to the model's own reasons (errors are common in the `arguments` field), causing JSON parsing to fail. These calls are stored in the `invalid_tool_calls` field. `ToolCallRepairMiddleware` will automatically detect `invalid_tool_calls` after the model returns results and attempt to repair them using `json-repair`, allowing tool calls to execute normally.

Please ensure that `langchain-dev-utils[standard]` is installed, as detailed in the **Installation Guide**.

This middleware is zero-configuration and ready to use out of the box, without additional parameters.

**Usage Example**:

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
    This middleware cannot guarantee 100% repair of all invalid tool calls. The actual effect depends on the repair capability of `json-repair`; additionally, it only acts on invalid tool call content in the `invalid_tool_calls` field.

## Formatting System Prompts

This middleware `format_prompt` allows you to use `f-string` style placeholders (such as `{name}`) in `system_prompt`, and dynamically replace them with actual values at runtime.

The values of variables in placeholders follow a clear parsing order:

1.  **First, look up in `state`**: It will first look for fields with the same name as the placeholder in the `state` dictionary.
2.  **Then, look up in `context`**: If the field is not found in `state`, it will continue to look in the `context` object.

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
        system_prompt="You are an intelligent assistant, your name is {name}. Your user is {user}.",
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
        system_prompt="You are an intelligent assistant, your name is {name}. Your user is {user}.",
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

    # The final system prompt will be "You are an intelligent assistant, your name is assistant-1. Your user is Zhang San."
    # Because state has higher priority
    print(response)
    ```

!!! warning "Note"
    There are two ways to implement custom middleware: decorators or class inheritance.  
    - Class inheritance implementation: `PlanMiddleware`, `ModelMiddleware`, `ToolCallRepairMiddleware`  
    - Decorator implementation: `format_prompt` (the decorator directly turns the function into a middleware instance, so no manual instantiation is needed)

!!! info "Note"
    In addition, this library has expanded the following middleware to support specifying models through string parameters:

    - SummarizationMiddleware

    - LLMToolSelectorMiddleware

    - ModelFallbackMiddleware
    
    - LLMToolEmulator

    You just need to import these middleware from this library to use strings to specify models that have been registered with `register_model_provider`. The middleware usage remains consistent with the official middleware, for example:
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