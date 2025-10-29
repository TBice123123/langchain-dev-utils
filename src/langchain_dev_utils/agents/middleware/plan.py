import json
from typing import Awaitable, Callable, Literal, Optional
from typing import NotRequired

from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelCallResult,
)
from langchain.tools import BaseTool, ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from typing_extensions import TypedDict

_DEFAULT_WRITE_PLAN_TOOL_DESCRIPTION = """
This tool is used to **create a new plan list ** or **modify an existing plan**.

Parameters:
`plan`: A list of strings representing the plans to be executed. The first plan in the list will automatically be set to the "in_progress" status, and all remaining plans will be set to the "pending" status. Internally, each string is converted into a dictionary containing "content" and "status" keys, where "content" corresponds to each string value in the list, and the "status" is assigned according to the rule: the first plan is "in_progress", and all others are "pending".
When modifying a plan, only provide the plans that need to be executed next. For example, if the current plan is ['Plan 1', 'Plan 2', 'Plan 3'], and Plan 1 is completed, but you determine that Plan 2 and Plan 3 are no longer suitable and should be replaced with ['Plan 4', 'Plan 5'], then only pass ['Plan 4', 'Plan 5'].

## Tool Functionality

This tool is used to **create a new plan list** or **modify an existing plan list**.

- **Primary Use**: Create a new plan list.
- **Secondary Use**: Modify the plan when problems are identified with the existing one.
- **Not For**: Marking plans as completed.

## Plan Statuses

Use the following statuses to track progress:

- `pending`: Plan has not yet started.
- `in_progress`: Currently being worked on.
- `done`: Plan has been completed.

## Usage Guidelines

### Creating a Plan (Primary Use)

Call this tool to create a structured plan when dealing with complex tasks:

- Complex multi-step tasks (≥3 steps).
- Non-trivial tasks requiring careful planning.
- The user explicitly requests a plan list.

### Modifying a Plan (Special Cases)

Modify an existing plan only under the following circumstances:

- The current plan structure is found to be problematic.
- The plan structure or content needs adjustment.
- The existing plan cannot be executed effectively.

**Important Update Behavior**: When updating a plan, only pass the list of plans that need to be executed next. The tool automatically handles status assignment – the first plan is set to `in_progress`, and the rest are set to `pending`.

### Plan Completion

**Important Note**: When completing a plan, call the `finish_sub_plan()` function to update the plan status. **Do not use this tool**.

## When to Use

**Appropriate Scenarios**:

- When starting a new complex work session.
- When needing to reorganize the plan structure.
- When the current plan needs revision based on new information.

**Scenarios to Avoid**:

- Simple tasks (<3 steps).
- When only needing to mark a plan as complete.
- Purely informational queries or conversation.
- Trivial tasks that can be completed directly.

## Best Practices

1. When creating a plan, the first plan is automatically set to `in_progress`.
2. Ensure sub-plans within the plan are specific and actionable.
3. Break down complex sub-plans into smaller steps.
4. Always keep at least one plan in the `in_progress` status unless all plans are completed.
5. When modifying a plan, only provide the sub-plans that need to be executed next.
6. Use clear, descriptive names for sub-plans.

## Internal Processing Logic

The tool automatically converts the input strings into a structured format:

- Input: `["Sub-plan 1", "Sub-plan 2", "Sub-plan 3"]`
- Internal Representation:
```json
  [
  {"content": "Sub-plan 1", "status": "in_progress"},
  {"content": "Sub-plan 2", "status": "pending"},
  {"content": "Sub-plan 3", "status": "pending"}
  ]
```

Please remember:

- For simple plans, execute them directly; there is no need to call this tool.
- Use the `finish_sub_plan()` function to mark sub-plans as completed.
- When modifying a plan, only provide the sub-plans that need to be executed next.
"""

_DEFAULT_FINISH_SUB_PLAN_TOOL_DESCRIPTION = """
This tool is used to mark the completion status of a sub-plan in an existing plan.

Functionality:
- Marks the sub-plan with status 'in_progress' as 'done'
- Sets the first sub-plan with status 'pending' to 'in_progress' (if one exists)

## Tool Purpose
Specifically designed to **mark the current sub-plan as completed** in an existing plan.

### When to Use
Use only when the current sub-plan is confirmed complete.

### Automatic Status Management
- `in_progress` → `done`
- First `pending` → `in_progress` (if any)

## Usage Example
Current plan status:
```json
[
    {"content": "Research market trends", "status": "done"},
    {"content": "Analyze competitor data", "status": "in_progress"},
    {"content": "Prepare summary report", "status": "pending"}
]
```

After calling finish_sub_plan():
```json
[
    {"content": "Research market trends", "status": "done"},
    {"content": "Analyze competitor data", "status": "done"},
    {"content": "Prepare summary report", "status": "in_progress"}
]
```

Remember:
- Only for marking completion—do not use to create or modify plans (use write_plan instead)
- Ensure the sub-plan is truly complete before calling
- No parameters needed; status transitions are handled automatically
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

    This function creates a tool that allows agents to update the status of sub-plans in a plan. Sub-plans can be marked as "done" to track progress.

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


_READ_PLAN_SYSTEM_PROMPT = """### 3. read_plan: View Current Plan
- **Purpose**: Retrieve the full current plan list with statuses, especially when you forget which sub-plan you're supposed to execute next.
- **No parameters required**—returns a complete snapshot of the active plan.
"""
_PLAN_MIDDLEWARE_SYSTEM_PROMPT = """
You can manage task plans using the following {num} tools:

## 1. write_plan: Create or Replace a Plan
- **Primary Purpose**: Generate a structured execution framework for complex tasks, or completely replace the remaining plan sequence when the current plan is no longer valid.
- **When to Use**:
  - The task requires 3 or more distinct, actionable steps
  - The user explicitly requests a plan list or "to-do plan"
  - The user provides multiple plans (e.g., numbered list, comma-separated items)
  - Execution reveals fundamental flaws in the current plan, requiring a new direction
- **Input Format**: A list of plan description strings, e.g., ["Analyze user needs", "Design system architecture", "Implement core module"]
- **Automatic Status Assignment**:
  - First plan → `"in_progress"`
  - All subsequent plans → `"pending"`
- **Plan Replacement Rule**: Provide **only the new plans that should be executed next**. Do not include completed, obsolete, or irrelevant plans.
- **Plan Quality Requirements**:
  - Plans must be specific, actionable, and verifiable
  - Break work into logical phases (chronological or dependency-based)
  - Define clear milestones and deliverable standards
  - Avoid vague, ambiguous, or non-executable descriptions
- **Do NOT Use When**:
  - The task is simple (<3 steps)
  - The request is conversational, informational, or a one-off query
  - You only need to mark a plan as complete (use `finish_sub_plan` instead)

## 2. finish_sub_plan: Mark Current Plan as Complete
- **Primary Purpose**: Confirm the current `"in_progress"` plan is fully done, mark it as `"done"`, and automatically promote the first `"pending"` plan to `"in_progress"`.
- **Call Only If ALL Conditions Are Met**:
  - The sub-plan has been **fully executed**
  - All specified requirements have been satisfied
  - There are no unresolved errors, omissions, or blockers
  - The output meets quality standards and has been verified
- **Automatic Behavior**:
  - No parameters needed—status transitions are handled internally
  - If no `"pending"` plans remain, the plan ends naturally
- **Never Call If**:
  - The plan is partially complete
  - Known issues or defects remain
  - Execution was blocked due to missing resources or dependencies
  - The result fails to meet expected quality

{read_plan_system_prompt}

## Plan Status Rules (Only These Three Are Valid)
- **`"pending"`**: Plan not yet started
- **`"in_progress"`**: Currently being executed (exactly one allowed at any time)
- **`"done"`**: Fully completed and verified  
> ⚠️ No other status values (e.g., "completed", "failed", "blocked") are permitted.

## General Usage Principles
1. **Execute simple plans directly**: If a request can be fulfilled in 1–2 steps, do not create a plan—just complete it.
2. **Decompose thoughtfully**: Break complex work into clear, independent, trackable sub-plans.
3. **Manage status rigorously**:
   - Always maintain exactly one `"in_progress"` plan while work is ongoing
   - Call `finish_sub_plan` immediately after plan completion—never delay
4. **Plan modification = full replacement**: Never edit individual plans. To adjust the plan, use `write_plan` with a new list of remaining plans.
5. **Respect user intent**: If the user explicitly asks for a plan—even for a simpler task—honor the request and create one.
"""


class PlanMiddleware(AgentMiddleware):
    """Middleware that provides plan management capabilities to agents.

    This middleware adds a `write_plan` and `finish_sub_plan` (and `read_plan` optional) tool that allows agents to create and manage
    structured plan lists for complex multi-step operations. It's designed to help
    agents track progress, organize complex tasks, and provide users with visibility
    into task completion status.

    The middleware automatically injects system prompts that guide the agent on how to use the plan functionality effectively.

    Args:
        system_prompt: Custom system prompt to guide the agent on using the plan tool.
            If not provided, uses the default `_PLAN_MIDDLEWARE_SYSTEM_PROMPT`.
        tools: List of tools to be added to the agent. The tools must be created by `create_write_plan_tool`, `create_finish_sub_plan_tool`, and `create_read_plan_tool`(optional).

    Example:
        ```python
        from langchain_dev_utils.agents.middleware.plan import PlanMiddleware
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
        tools: Optional[list[BaseTool]] = None,
    ) -> None:
        super().__init__()

        if tools is None:
            tools = [
                create_write_plan_tool(),
                create_finish_sub_plan_tool(),
                create_read_plan_tool(),
            ]

        # Check if required tools exist
        has_write_plan = any(tool_obj.name == "write_plan" for tool_obj in tools)
        has_finish_sub_plan = any(
            tool_obj.name == "finish_sub_plan" for tool_obj in tools
        )

        if not (has_write_plan and has_finish_sub_plan):
            raise ValueError(
                "PlanMiddleware requires exactly two tools: write_plan and finish_sub_plan."
            )

        self.tools = tools

        if system_prompt is None:
            num = len(self.tools)
            read_plan_system = (
                _READ_PLAN_SYSTEM_PROMPT if num == 3 else ""
            )  # if read_plan tool is not provided, do not include it in the system prompt

            system_prompt = _PLAN_MIDDLEWARE_SYSTEM_PROMPT.format(
                num=num, read_plan_system_prompt=read_plan_system
            )

        self.system_prompt = system_prompt

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        request.system_prompt = (
            request.system_prompt + "\n\n" + self.system_prompt
            if request.system_prompt
            else self.system_prompt
        )
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        request.system_prompt = (
            request.system_prompt + "\n\n" + self.system_prompt
            if request.system_prompt
            else self.system_prompt
        )
        return await handler(request)
