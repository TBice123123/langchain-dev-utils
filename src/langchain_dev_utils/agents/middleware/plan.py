import json
from typing import Awaitable, Callable, Literal, NotRequired, Optional, cast

from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelCallResult,
)
from langchain.tools import BaseTool, ToolRuntime, tool
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.types import Command
from typing_extensions import TypedDict

_DEFAULT_WRITE_PLAN_TOOL_DESCRIPTION = """Use this tool to create and manage a structured task list for complex or multi-step work. It helps you stay organized, track progress, and demonstrate to the user that you’re handling tasks systematically.

## When to Use This Tool  
Use this tool in the following scenarios:

1. **Complex multi-step tasks** — when a task requires three or more distinct steps or actions.  
2. **Non-trivial and complex tasks** — tasks that require careful planning or involve multiple operations.  
3. **User explicitly requests a to-do list** — when the user directly asks you to use the to-do list feature.  
4. **User provides multiple tasks** — when the user supplies a list of items to be done (e.g., numbered or comma-separated).  
5. **The plan needs adjustment based on current execution** — when ongoing progress indicates the plan should be revised.

## How to Use This Tool  
1. **When starting a task** — before actually beginning work, invoke this tool with a task list (a list of strings). The first task will automatically be set to `in_progress`, and all others to `pending`.  
2. **When updating the task list** — for example, after completing some tasks, if you find certain tasks are no longer needed, remove them; if new necessary tasks emerge, add them. However, **do not modify** tasks already marked as completed. In such cases, simply call this tool again with the updated task list.

## When NOT to Use This Tool  
Avoid using this tool in the following situations:  
1. The task is a **single, straightforward action**.  
2. The task is **too trivial**, and tracking it provides no benefit.  
3. The task can be completed in **fewer than three simple steps**.  
4. The current task list has been fully completed — in this case, use `finish_sub_plan()` to finalize.

## How It Works  
- **Input**: A parameter named `plan` containing a list of strings representing the tasks (e.g., `["Task 1", "Task 2", "Task 3"]`).  
- **Automatic status assignment**:  
  → First task: `in_progress`  
  → Remaining tasks: `pending`  
- When updating the plan, provide only the **next set of tasks to execute**. For example, if the next phase requires `["Task 4", "Task 5"]`, call this tool with `plan=["Task 4", "Task 5"]`.

## Task States  
- `pending`: Ready to start, awaiting execution  
- `in_progress`: Currently being worked on  
- `done`: Completed  

## Best Practices  
- Break large tasks into clear, actionable steps.  
- Use specific and descriptive task names.  
- Update the plan immediately if priorities shift or blockers arise.  
- Never leave the plan empty — as long as unfinished tasks remain, at least one must be marked `in_progress`.  
- Do not batch completions — mark each task as done immediately after finishing it.  
- Remove irrelevant tasks entirely instead of leaving them in `pending` state.

**Remember**: If a task is simple, just do it. This tool is meant to provide structure — not overhead.
"""

_DEFAULT_FINISH_SUB_PLAN_TOOL_DESCRIPTION = """This tool is used to mark the currently in-progress task in an existing task list as completed.

## Functionality  
- Marks the current task with status `in_progress` as `done`, and automatically sets the next task (previously `pending`) to `in_progress`.

## When to Use  
Use only when you have confirmed that the current task is truly finished.

## Example  
Before calling:
```json
[
    {"content": "Task 1", "status": "done"},
    {"content": "Task 2", "status": "in_progress"},
    {"content": "Task 3", "status": "pending"}
]
```

After calling `finish_sub_plan()`:
```json
[
    {"content": "Task 1", "status": "done"},
    {"content": "Task 2", "status": "done"},
    {"content": "Task 3", "status": "in_progress"}
]
```

**Note**:  
- This tool is **only** for marking completion — do **not** use it to create or modify plans (use `write_plan` instead).  
- Ensure the task is genuinely complete before invoking this function.  
- No parameters are required — status updates are handled automatically.
"""

_DEFAULT_READ_PLAN_TOOL_DESCRIPTION = """
Get all sub-plans with their current status.
"""


class Plan(TypedDict):
    content: str
    status: Literal["pending", "in_progress", "done"]


class PlanState(AgentState):
    plan: NotRequired[list[Plan]]


def create_write_plan_tool(
    description: Optional[str] = None,
    message_key: Optional[str] = None,
) -> BaseTool:
    """Create a tool for writing initial plan.

    This function creates a tool that allows agents to write an initial plan
    with a list of plans. The first plan in the plan will be marked as "in_progress"
    and the rest as "pending".

    Args:
        description: The description of the tool. Uses default description if not provided.
        message_key: The key of the message to be updated. Defaults to "messages".

    Returns:
        BaseTool: The tool for writing initial plan.

    Example:
        Basic usage:
        >>> from langchain_dev_utils.agents.middleware import create_write_plan_tool
        >>> write_plan_tool = create_write_plan_tool()
    """

    @tool(
        description=description or _DEFAULT_WRITE_PLAN_TOOL_DESCRIPTION,
    )
    def write_plan(plan: list[str], runtime: ToolRuntime):
        msg_key = message_key or "messages"
        return Command(
            update={
                "plan": [
                    {
                        "content": content,
                        "status": "pending" if index > 0 else "in_progress",
                    }
                    for index, content in enumerate(plan)
                ],
                msg_key: [
                    ToolMessage(
                        content=f"Plan successfully written, please first execute the {plan[0]} sub-plan (no need to change the status to in_process)",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    return write_plan


def create_finish_sub_plan_tool(
    description: Optional[str] = None,
    message_key: Optional[str] = None,
) -> BaseTool:
    """Create a tool for finishing sub-plan tasks.

    This function creates a tool that allows agents to update the status of sub-plans in a plan.
    Sub-plans can be marked as "done" to track progress.

    Args:
        description: The description of the tool. Uses default description if not provided.
        message_key: The key of the message to be updated. Defaults to "messages".

    Returns:
        BaseTool: The tool for finishing sub-plan tasks.

    Example:
        Basic usage:
        >>> from langchain_dev_utils.agents.middleware import create_finish_sub_plan_tool
        >>> finish_sub_plan_tool = create_finish_sub_plan_tool()
    """

    @tool(
        description=description or _DEFAULT_FINISH_SUB_PLAN_TOOL_DESCRIPTION,
    )
    def finish_sub_plan(
        runtime: ToolRuntime,
    ):
        msg_key = message_key or "messages"
        plan_list = runtime.state.get("plan", [])

        sub_finish_plan = ""
        sub_next_plan = ",all sub plan are done"
        for plan in plan_list:
            if plan["status"] == "in_progress":
                plan["status"] = "done"
                sub_finish_plan = f"finish sub plan:**{plan['content']}**"

        for plan in plan_list:
            if plan["status"] == "pending":
                plan["status"] = "in_progress"
                sub_next_plan = f",next plan:**{plan['content']}**"
                break

        return Command(
            update={
                "plan": plan_list,
                msg_key: [
                    ToolMessage(
                        content=sub_finish_plan + sub_next_plan,
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    return finish_sub_plan


def create_read_plan_tool(
    description: Optional[str] = None,
):
    """Create a tool for reading all sub-plans.

    This function creates a tool that allows agents to read all sub-plans
    in the current plan with their status information.

    Args:
        description: The description of the tool. Uses default description if not provided.

    Returns:
        BaseTool: The tool for reading all sub-plans.

    Example:
        Basic usage:
        >>> from langchain_dev_utils.agents.middleware import create_read_plan_tool
        >>> read_plan_tool = create_read_plan_tool()
    """

    @tool(
        description=description or _DEFAULT_READ_PLAN_TOOL_DESCRIPTION,
    )
    def read_plan(runtime: ToolRuntime):
        plan_list = runtime.state.get("plan", [])
        return json.dumps(plan_list)

    return read_plan


_PLAN_SYSTEM_PROMPT_NOT_READ_PLAN = """You can manage task plans using two simple tools:

## write_plan
- Use it to break complex tasks (3+ steps) into a clear, actionable list. Only include next steps to execute — the first becomes `"in_progress"`, the rest `"pending"`. Don’t use it for simple tasks (<3 steps).

## finish_sub_plan
- Call it **only when the current task is 100% done**. It automatically marks it `"done"` and promotes the next `"pending"` task to `"in_progress"`. No parameters needed. Never use it mid-task or if anything’s incomplete.
Keep plans lean, update immediately, and never batch completions.

**Note**: Make sure that all tasks end up with the status `"done"`.
"""

_PLAN_SYSTEM_PROMPT = """You can manage task plans using three simple tools:

## write_plan
- Use it to break complex tasks (3+ steps) into a clear, actionable list. Only include next steps to execute — the first becomes `"in_progress"`, the rest `"pending"`. Don’t use it for simple tasks (<3 steps).

## finish_sub_plan
- Call it **only when the current task is 100% done**. It automatically marks it `"done"` and promotes the next `"pending"` task to `"in_progress"`. No parameters needed. Never use it mid-task or if anything’s incomplete.

## read_plan
- Retrieve the full current plan list with statuses, especially when you forget which sub-plan you're supposed to execute next.
- No parameters required—returns a current plan list with statuses.

**Note**: Make sure that all tasks end up with the status `"done"`.
"""


class PlanMiddleware(AgentMiddleware):
    """Middleware that provides plan management capabilities to agents.

    This middleware adds a `write_plan` and `finish_sub_plan` (and `read_plan`
    optional) tool that allows agents to create and manage structured plan lists
    for complex multi-step operations. It's designed to help agents track progress,
    organize complex tasks, and provide users with visibility into task completion
    status.

    The middleware automatically injects system prompts that guide the agent on
    how to use the plan functionality effectively.

    Args:
        system_prompt: Custom system prompt to guide the agent on using the plan
            tool. If not provided, uses the default `_PLAN_SYSTEM_PROMPT` or
            `_PLAN_SYSTEM_PROMPT_NOT_READ_PLAN` based on the `use_read_plan_tool`
            parameter.
        write_plan_tool_description: Description of the `write_plan` tool.
            If not provided, uses the default `_DEFAULT_WRITE_PLAN_TOOL_DESCRIPTION`.
        finish_sub_plan_tool_description: Description of the `finish_sub_plan` tool.
            If not provided, uses the default `_DEFAULT_FINISH_SUB_PLAN_TOOL_DESCRIPTION`.
        read_plan_tool_description: Description of the `read_plan` tool.
            If not provided, uses the default `_DEFAULT_READ_PLAN_TOOL_DESCRIPTION`.
        use_read_plan_tool: Whether to use the `read_plan` tool.
            If not provided, uses the default `True`.
        message_key: The key of the message to be updated. Defaults to "messages".
    Example:
        ```python
        from langchain_dev_utils.agents.middleware import PlanMiddleware
        from langchain_dev_utils.agents import create_agent

        agent = create_agent("vllm:qwen3-4b", middleware=[PlanMiddleware()])

        # Agent now has access to write_plan tool and plan state tracking
        result = await agent.invoke({"messages": [HumanMessage("Help me refactor my codebase")]})

        print(result["plan"])  # Array of plan items with status tracking
        ```
    """

    state_schema = PlanState

    def __init__(
        self,
        *,
        system_prompt: Optional[str] = None,
        write_plan_tool_description: Optional[str] = None,
        finish_sub_plan_tool_description: Optional[str] = None,
        read_plan_tool_description: Optional[str] = None,
        use_read_plan_tool: bool = True,
        message_key: Optional[str] = None,
    ) -> None:
        super().__init__()

        write_plan_tool_description = (
            write_plan_tool_description or _DEFAULT_WRITE_PLAN_TOOL_DESCRIPTION
        )
        finish_sub_plan_tool_description = (
            finish_sub_plan_tool_description
            or _DEFAULT_FINISH_SUB_PLAN_TOOL_DESCRIPTION
        )
        read_plan_tool_description = (
            read_plan_tool_description or _DEFAULT_READ_PLAN_TOOL_DESCRIPTION
        )

        tools = [
            create_write_plan_tool(
                description=write_plan_tool_description, message_key=message_key
            ),
            create_finish_sub_plan_tool(
                description=finish_sub_plan_tool_description, message_key=message_key
            ),
        ]

        if use_read_plan_tool:
            tools.append(create_read_plan_tool(description=read_plan_tool_description))

        if system_prompt is None:
            if use_read_plan_tool:
                system_prompt = _PLAN_SYSTEM_PROMPT
            else:
                system_prompt = _PLAN_SYSTEM_PROMPT_NOT_READ_PLAN

        self.system_prompt = system_prompt
        self.tools = tools

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Update the system message to include the plan system prompt."""
        if request.system_message is not None:
            new_system_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": f"\n\n{self.system_prompt}"},
            ]
        else:
            new_system_content = [{"type": "text", "text": self.system_prompt}]
        new_system_message = SystemMessage(
            content=cast("list[str | dict[str, str]]", new_system_content)
        )
        return handler(request.override(system_message=new_system_message))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Update the system message to include the plan system prompt."""
        if request.system_message is not None:
            new_system_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": f"\n\n{self.system_prompt}"},
            ]
        else:
            new_system_content = [{"type": "text", "text": self.system_prompt}]
        new_system_message = SystemMessage(
            content=cast("list[str | dict[str, str]]", new_system_content)
        )
        return await handler(request.override(system_message=new_system_message))
