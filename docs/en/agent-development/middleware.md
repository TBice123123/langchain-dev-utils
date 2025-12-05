# Middleware

> [!NOTE]
>
> **Function Overview**: Provides practical tools for convenient Agent development.
>
> **Prerequisites**: Understanding of langchain's [middleware](https://docs.langchain.com/oss/python/langchain/middleware).
>
> **Estimated Reading Time**: 12 minutes

## Overview

Middleware is a component specifically built for pre-built `langchain` Agents. The official documentation provides some built-in middleware. This library, based on actual needs and usage scenarios, offers additional middleware.

## Task Planning

Task planning middleware is used for structured decomposition and process management before executing complex tasks.

::: tip 📝
Task planning is an efficient context engineering management strategy. Before executing a task, the large model first breaks down the overall task into multiple ordered subtasks, forming a task planning list (called "plan" in this library). Then it executes each subtask in sequence, dynamically updating the task status after completing each step, until all subtasks are completed.
:::

The middleware that implements task planning is `PlanMiddleware`, with the following parameter descriptions:

<Params
name="system_prompt"
type="string"
description="Optional string type, system prompt."
:required="false"
:default="null"
/>
<Params 
name="write_plan_tool_description"
type="string"
description="Optional string type, description of the write plan tool."
:required="false"
:default="null"
/>
<Params 
name="finish_sub_plan_tool_description"
type="string"
description="Optional string type, description of the finish sub plan tool."
:required="false"
:default="null"
/>
<Params 
name="read_plan_tool_description"
type="string"
description="Optional string type, description of the read plan tool."
:required="false"
:default="null"
/>
<Params 
name="use_read_plan_tool"
type="bool"
description="Optional boolean type, whether to use the read plan tool."
:required="false"
:default="true"
/>

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

`PlanMiddleware` requires the use of two tools: `write_plan` and `finish_sub_plan`, while the `read_plan` tool is enabled by default; if not needed, you can set the `use_read_plan_tool` parameter to `False`.

This middleware is similar in functionality to the **To-do list middleware** provided by LangChain officially, but there are differences in tool design. The official middleware only provides the `write_todo` tool, which is oriented towards a todo list structure; while this library provides three specialized tools: `write_plan`, `finish_sub_plan`, and `read_plan`, specifically for writing, modifying, and querying plan lists.

Whether it's `todo` or `plan`, their essence is the same, so the key difference between this middleware and the official one lies in the tools provided. The official addition and modification are completed through one tool, while this library provides three tools, where `write_plan` can be used to write or update plan content, `finish_sub_plan` is used to update the status after completing a subtask, and `read_plan` is used to query plan content.

At the same time, this library also provides three functions to create the above three tools:

- `create_write_plan_tool`: A function to create a tool for writing plans
- `create_finish_sub_plan_tool`: A function to create a tool for completing subtasks
- `create_read_plan_tool`: A function to create a tool for querying plans

The parameters received by these three functions are as follows:

<Params
name="description"
type="string"
description="Tool description. If not passed, the default tool description is used."
:required="false"
:default="null"
/>
<Params
name="message_key"
type="string"
description="The key for updating messages. If not passed, the default messages is used (read_plan tool has no this parameter)."
:required="false"
:default="null"
/>

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

<BestPractice>

I. When using <code>create_agent</code>:

It is recommended to directly use <code>PlanMiddleware</code> instead of manually passing in the three tools <code>write_plan</code>, <code>finish_sub_plan</code>, and <code>read_plan</code>.

Reason: The middleware has automatically handled prompt construction and agent state management, which can significantly reduce usage complexity.

Note: Since the model output of <code>create_agent</code> is fixed to update to the <code>messages</code> key, <code>PlanMiddleware</code> does not have a <code>message_key</code> parameter.

II. When using <code>langgraph</code>:

It is recommended to directly use these three tools (<code>write_plan</code>, <code>finish_sub_plan</code>, <code>read_plan</code>).

Reason: This approach can better integrate with <code>langgraph</code>'s custom nodes and state management.
</BestPractice>

## Model Routing

`ModelRouterMiddleware` is a middleware for **dynamically routing to the most suitable model based on input content**. It analyzes user requests through a "router model" and selects the most suitable model from a predefined list of models to handle the current task.

Its parameters are as follows:

<Params
name="router_model"
type="BaseChatModel | string"
description="The model used to execute routing decisions. You can pass a string (which will be automatically loaded through load_chat_model), such as vllm:qwen3-4b; or directly pass an instantiated BaseChatModel object."
:required="true"
:default="null"
/>
<Params
name="model_list"
type="list[dict]"
description="A list of model configurations, each element is a dictionary, which needs to include model_name (str), model_description (str), and optional tools (list[BaseTool]), model_kwargs (dict), model_system_prompt (str)."
:required="true"
:default="null"
/>
<Params
name="router_prompt"
type="string"
description="Custom prompt for the router model. If None (default), the built-in default prompt template is used."
:required="false"
:default="null"
/>

**Usage Example**

First define the model list:

```python
model_list = [
    {
        "model_name": "vllm:qwen3-8b",
        "model_description": "Suitable for general tasks, such as dialogue, text generation, etc.",
        "model_kwargs": {
            "temperature": 0.7,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
        },
        "model_system_prompt": "You are an assistant, good at handling general tasks, such as dialogue, text generation, etc.",
    },
    {
        "model_name": "openrouter:qwen/qwen3-vl-32b-instruct",
        "model_description": "Suitable for visual tasks",
        "tools": [],  # If this model does not need any tools, set this field to an empty list []
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

Through `ModelRouterMiddleware`, you can easily build a multi-model, multi-capability Agent that automatically selects the optimal model according to the task type, improving response quality and efficiency.

::: info Tool Permission Configuration  
The tool permissions for each model in `model_list` are determined by the configuration of its `tools` field, which follows the following rules:

- **When undefined**: The model inherits all tools loaded by the `create_agent` parameter `tools`.
- **Defined as an empty list []**: The model is explicitly disabled from all tools.
- **Defined as a non-empty list [tool1, tool2, ...]**: This list acts as a "tool whitelist", and the model is strictly limited to only call tools within the list. All tools specified here must have been preloaded into the `create_agent` parameter `tools`.

:::

## Tool Call Repair
`ToolCallRepairMiddleware` is a middleware for **automatically repairing invalid tool calls (`invalid_tool_calls`) by large models**.

When large models output JSON Schema for tool calls, they may generate JSON format errors due to the model itself (error content is common in the `arguments` field), causing JSON parsing to fail. Such calls will be stored in the `invalid_tool_calls` field. `ToolCallRepairMiddleware` will automatically detect `invalid_tool_calls` after the model returns results, and attempt to call `json-repair` for repair, so that tool calls can be executed normally.

Make sure you have installed `langchain-dev-utils[standard]`, see [Installation Guide](../installation.md) for details.

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

::: warning Note
This middleware cannot guarantee 100% repair of all invalid tool calls. The actual effect depends on the repair capability of `json-repair`; additionally, it only acts on invalid tool call content in the `invalid_tool_calls` field.
:::


::: info Note
In addition, this library has also expanded the following middleware to support the function of specifying models through string parameters:
- SummarizationMiddleware
- LLMToolSelectorMiddleware
- ModelFallbackMiddleware
- LLMToolEmulator

You just need to import these middleware from this library to use strings to specify models that have been registered by `register_model_provider`. The usage of middleware is consistent with the official middleware, for example:
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

:::