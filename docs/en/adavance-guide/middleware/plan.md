# Task Planning

`PlanMiddleware` is a middleware used for structured decomposition and process management before executing complex tasks.

!!! info "Additional Notes"

    Task planning is an efficient strategy for context engineering management. Before executing a task, the large language model first breaks down the overall task into multiple ordered subtasks, forming a task planning list (referred to as a plan in this library). It then executes each subtask sequentially, dynamically updating the task status after completing each step, until all subtasks are finished.

## Parameter Description

| Parameter | Description |
|------|------|
| `system_prompt` | System prompt. If `None`, the default prompt will be used.<br><br>**Type**: `str`<br>**Required**: No |
| `custom_plan_tool_descriptions` | Custom descriptions for planning-related tools.<br><br>**Type**: `dict`<br>**Required**: No |
| `use_read_plan_tool` | Whether to enable the read plan tool.<br><br>**Type**: `bool`<br>**Required**: No<br>**Default Value**: `True` |

The keys for the `custom_plan_tool_descriptions` dictionary can be any of the following three values:

| Key | Description |
|------|------|
| `write_plan` | Description for the write plan tool |
| `finish_sub_plan` | Description for the finish sub-plan tool |
| `read_plan` | Description for the read plan tool |

## Usage Example

```python
from langchain_dev_utils.agents.middleware import PlanMiddleware

agent = create_agent(
    model="vllm:qwen3-4b",
    middleware=[
        PlanMiddleware(
            custom_plan_tool_descriptions={
                "write_plan": "Used to write a plan, breaking down the task into multiple ordered subtasks.",
                "finish_sub_plan": "Used to complete a subtask, updating its status to finished.",
                "read_plan": "Used to query the current task planning list."
            },
            use_read_plan_tool=True,  # If you don't want to use the read plan tool, you can set this parameter to False
        )
    ],
)

response = agent.invoke(
    {"messages": [HumanMessage(content="I'm going to New York for a few days, help me plan the itinerary")]}
)
print(response)
```

## Tool Description

`PlanMiddleware` requires the use of two mandatory tools: `write_plan` and `finish_sub_plan`, while the `read_plan` tool is enabled by default. If not needed, you can set the `use_read_plan_tool` parameter to `False`.

## Comparison with the Official To-do List Middleware

This middleware is similar in function to the official **To-do list middleware** provided by LangChain, but differs in tool design:

| Feature | Official To-do List Middleware | This Library's PlanMiddleware |
|------|----------------------|---------------------|
| Number of Tools | 1 (`write_todo`) | 3 (`write_plan`, `finish_sub_plan`, `read_plan`) |
| Functionality Focus | Oriented towards to-do lists | Specifically designed for planning lists |
| Operation Method | Adding and modifying are done through one tool | Writing, modifying, and querying are handled by separate tools |

Whether referred to as `todo` or `plan`, the essence is the same concept. The key distinction of this middleware from the official one is that it provides three dedicated tools:

- `write_plan`: Used to write or update plan content
- `finish_sub_plan`: Used to update the status of a subtask after its completion
- `read_plan`: Used to query plan content