# Format System Prompts

Placeholder variables in system prompts can be dynamically resolved through middleware. This library provides two implementations:

1.  **Global Instance `format_prompt`**: Pre-configured with **f-string** style formatting logic (e.g., `{name}`). Suitable for most scenarios and recommended for direct use.
2.  **Middleware Class `FormatPromptMiddleware`**: This class must be manually instantiated when **jinja2** style formatting (e.g., `{{ name }}`) is required.

!!! warning "Usage Notes"
    Ensure you have installed `langchain-dev-utils[standard]` if you need to use **jinja2** style templates. Please refer to the **Installation Guide**.

## Variable Resolution Order

The values for placeholder variables are resolved following a priority order from highest to lowest:

1.  **First, check the `state`**: The `state` dictionary is searched for a field matching the placeholder name.
2.  **Then, check the `context`**: If the field is not found in `state`, the lookup continues in the `context` object.

This means values in `state` have higher priority and can override values with the same name in `context`.

## Using f-string Style (`format_prompt`)

`format_prompt` is the most commonly used global instance, employing Python's native f-string syntax. All examples below are based on this instance.

### Getting Variables Only from `state`

This is the most basic usage, where all placeholder variables are provided by `state`.

```python
from langchain_dev_utils.agents.middleware import format_prompt
from langchain.agents import AgentState

class AssistantState(AgentState):
    name: str

agent = create_agent(
    model="vllm:qwen2.5-7b",
    system_prompt="You are an intelligent assistant, and your name is {name}.",
    middleware=[format_prompt],
    state_schema=AssistantState,
)

# When invoking, the 'name' value must be provided in state
response = agent.invoke(
    {"messages": [HumanMessage(content="Hello there")], "name": "assistant"}
)
print(response)
```

### Getting Variables from Both `state` and `context`

The following example demonstrates how to mix data from both `state` and `context`:

```python
from dataclasses import dataclass

@dataclass
class Context:
    user: str

agent = create_agent(
    model="vllm:qwen2.5-7b",
    # {name} will be obtained from state, {user} will be obtained from context
    system_prompt="You are an intelligent assistant, and your name is {name}. Your user is named {user}.",
    middleware=[format_prompt],
    state_schema=AssistantState,
    context_schema=Context,
)

# When invoking, provide 'name' for state and 'user' for context
response = agent.invoke(
    {
        "messages": [HumanMessage(content="I'm going to New York for a few days, help me plan the itinerary")],
        "name": "assistant",
    },
    context=Context(user="Zhang San"),
)
print(response)
```

### Variable Override Example

When a variable with the same name exists in both `state` and `context`, the value from `state` takes precedence.

```python
from dataclasses import dataclass

@dataclass
class Context:
    # 'name' is defined in context
    name: str
    user: str

agent = create_agent(
    model="vllm:qwen2.5-7b",
    system_prompt="You are an intelligent assistant, and your name is {name}. Your user is named {user}.",
    middleware=[format_prompt],
    state_schema=AssistantState, # 'name' is also defined in state
    context_schema=Context,
)

# When invoking, values for 'name' are provided in both state and context
response = agent.invoke(
    {
        "messages": [HumanMessage(content="What is your name?")],
        "name": "assistant-1",
    },
    context=Context(name="assistant-2", user="Zhang San"),
)

# The final system prompt will be: "You are an intelligent assistant, and your name is assistant-1. Your user is named Zhang San."
# Because state has higher priority
print(response)
```

## Using Jinja2 Style (`FormatPromptMiddleware`)

Use `FormatPromptMiddleware` and specify `template_format="jinja2"` if your system prompts require more complex logic (such as loops, conditional statements) or if you prefer Jinja2 syntax.

### Basic Example

The example below demonstrates how to use Jinja2 syntax to dynamically generate prompts based on conditions.

```python
from langchain_dev_utils.agents.middleware import FormatPromptMiddleware
from dataclasses import dataclass
from typing import Optional

@dataclass
class Context:
    user_role: Optional[str] = None  # User role, e.g., "VIP", "Admin"

# Manually instantiate the middleware, specifying the format as jinja2
jinja2_formatter = FormatPromptMiddleware(template_format="jinja2")

agent = create_agent(
    model="vllm:qwen2.5-7b",
    # Use {{ }} syntax
    system_prompt=(
        "You are an intelligent assistant.\n"
        "{% if user_role == 'VIP' %}"
        "Please provide premium, attentive service.\n"
        "{% elif user_role == 'Admin' %}"
        "Demonstrate the authority and rigor of a system administrator.\n"
        "{% else %}"
        "Provide standard user service.\n"
        "{% endif %}"
    ),
    middleware=[jinja2_formatter],
    context_schema=Context,
)

# Example 1: Regular user
response = agent.invoke(
    {"messages": [HumanMessage(content="Hello")]},
    context=Context(user_role="Guest"),
)

# Example 2: VIP user
# The system prompt will contain "Please provide premium, attentive service."
response = agent.invoke(
    {"messages": [HumanMessage(content="Hello")]},
    context=Context(user_role="VIP"),
)
```

!!! warning "Jinja2 Template Caution"
    When Jinja2 templating is enabled, `system_prompt` will be compiled into a Template object. **Always hardcode the prompt skeleton in your codebase**, injecting dynamic values solely via `state` or `context`. **Never** pass raw user input directly as the `system_prompt` argument to `create_agent`. (Regardless of this feature, `system_prompt` should always be fully controlled by developers and never directly accept external user input as the core system prompt.)