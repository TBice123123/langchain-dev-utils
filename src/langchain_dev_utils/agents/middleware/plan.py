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
This tool is used to create new task plans or modify existing plans.

Args:
    task_list: A list of strings representing tasks to be executed.
               The first task will automatically be set to 'in_progress' status,
               while all other tasks will be set to 'pending' status.
               Internally, each string is converted to a dictionary containing 
               'content' and 'status' keys, where content corresponds to each value in the string list,
               and status values are: the first task is 'in_progress', and the remaining tasks are 'pending'.

               When modifying plans, only provide the tasks that need to be executed next.
               For example, if the current plan is 
               ['task1', 'task2', 'task3'] and task1 is completed, but you
               determine that task2 and task3 are no longer appropriate and
               should be replaced with ['task4', 'task5'], simply pass
               ['task4', 'task5'].

## Tool Function
This tool is used to **create new plans** or **modify existing plans**.

- **Primary use**: Create new task plans
- **Secondary use**: Modify plans when issues are identified in existing plans
- **Not for**: Marking task completion status

## Task States
Use the following states to track progress:
- `pending`: Task not yet started
- `in_progress`: Currently being processed
- `done`: Task completed

## Usage Specifications

### Creating Plans (Primary Use)
Call this tool to create structured task lists when dealing with complex tasks:
- Complex multi-step tasks (≥3 steps)
- Non-trivial tasks requiring careful planning
- When user explicitly requests a task list
- When user provides multiple task items

### Modifying Plans (Special Circumstances)
Modify existing plans only in the following cases:
- When current plan structure is identified as problematic
- When task structure or content needs adjustment
- When existing plan cannot be executed effectively

**Important Update Behavior**: When updating plans, only pass the list of tasks that need to be executed next. The tool will automatically handle status assignment - first task as `in_progress`, others as `pending`.

### Task Completion
**Important**: When completing subtasks, call the `finish_sub_plan()` function to update task status, NOT this tool.

## When to Use
**Usage scenarios**:
- When starting new complex work sessions
- When needing to restructure task organization
- When current plan requires revision based on new information

**Avoid usage scenarios**:
- Simple tasks (<3 steps)
- When only needing to mark task completion status
- Purely informational inquiries or conversations
- Trivial tasks that can be completed directly

## Best Practices
1. When creating a plan, the first task is automatically set to `in_progress`
2. Ensure tasks in the plan are specific and actionable
3. Break down complex tasks into smaller steps
4. Unless all tasks are completed, always maintain at least one task in `in_progress` status
5. When modifying plans, provide only the remaining tasks that need execution
6. Use descriptive and clear task names

## Internal Processing
The tool automatically converts input strings to structured format:
- Input: `["task1", "task2", "task3"]`
- Internal representation:
  [
    {"content": "task1", "status": "in_progress"},
    {"content": "task2", "status": "pending"},
    {"content": "task3", "status": "pending"}
  ]

Remember:
- For simple tasks, execute directly without calling this tool
- Use `finish_sub_plan()` function when completing tasks
- When modifying plans, only provide the tasks that need to be executed next
"""

_DEFAULT_FINISH_SUB_PLAN_TOOL_DESCRIPTION = """
This tool is used to mark the completion status of a subtask in an existing plan.

Functionality:
    - Marks the task with status 'in_progress' as 'done'
    - Sets the first task with status 'pending' to 'in_progress' (if one exists)

## Tool Purpose
Specifically designed to **mark the current subtask as completed** in an existing plan.

### When to Use
Use only when the current subtask is confirmed complete.

### Automatic Status Management
- `in_progress` → `done`
- First `pending` → `in_progress` (if any)

## Usage Example
Current plan status:
[
    {"content": "Research market trends", "status": "done"},
    {"content": "Analyze competitor data", "status": "in_progress"},
    {"content": "Prepare summary report", "status": "pending"}
]

After calling finish_sub_plan():
[
    {"content": "Research market trends", "status": "done"},
    {"content": "Analyze competitor data", "status": "done"},
    {"content": "Prepare summary report", "status": "in_progress"}
]

Remember:
- Only for marking completion—do not use to create or modify plans (use write_plan instead)
- Ensure the task is truly complete before calling
- No parameters needed; status transitions are handled automatically
"""

_DEFAULT_READ_PLAN_TOOL_DESCRIPTION = """
Retrieves the current task plan with all subtasks and their statuses.

### When to Use
When you forget which subtask you're currently supposed to execute.
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
    with a list of tasks. The first task in the plan will be marked as "in_progress"
    and the rest as "pending".

    Args:
        name: The name of the tool. Defaults to "write_plan".
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
                        content=f"Plan successfully written, please first execute the {plan[0]} task (no need to change the status to in_process)",
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
    """Create a tool for finish sub plan tasks.

    This function creates a tool that allows agents to update the status of tasks
    in a plan. Tasks can be marked as "done" to track progress.

    Args:
        name: The name of the tool. Defaults to "finish_sub_plan".
        description: The description of the tool. Uses default description if not provided.
        message_key: The key of the message to be updated. Defaults to "messages".

    Returns:
        BaseTool: The tool for finish sub plan tasks.

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
        plan_list = runtime.state.get("plan", [])

        sub_finish_plan = ""
        sub_next_plan = ""
        for plan in plan_list:
            if plan["status"] == "in_progress":
                plan["status"] = "done"
                sub_finish_plan = plan["content"]

        for plan in plan_list:
            if plan["status"] == "pending":
                plan["status"] = "in_progress"
                sub_next_plan = plan["content"]
                break

        return Command(
            update={
                "plan": plan_list,
                "messages": [
                    ToolMessage(
                        content=f"finish sub plan {sub_finish_plan}, next plan {sub_next_plan}",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    return finish_sub_plan


def create_read_plan_tool(
    description: Optional[str] = None,
):
    @tool(
        description=description or _DEFAULT_READ_PLAN_TOOL_DESCRIPTION,
    )
    def read_plan(runtime: ToolRuntime):
        plan_list = runtime.state.get("plan", [])
        return json.dumps(plan_list)

    return read_plan


_READ_PLAN_SYSTEM_PROMPT = """### 3. read_plan: View Current Plan
- **Purpose**: Retrieve the full current task list with statuses, especially when you forget which subtask you're supposed to execute next.
- **No parameters required**—returns a complete snapshot of the active plan.
"""
_PLAN_MIDDLEWARE_SYSTEM_PROMPT = """
You can manage task plans using the following {num} tools:

## 1. write_plan: Create or Replace a Plan
- **Primary Purpose**: Generate a structured execution framework for complex tasks, or completely replace the remaining task sequence when the current plan is no longer valid.
- **When to Use**:
  - The task requires 3 or more distinct, actionable steps
  - The user explicitly requests a task list or "to-do plan"
  - The user provides multiple tasks (e.g., numbered list, comma-separated items)
  - Execution reveals fundamental flaws in the current plan, requiring a new direction
- **Input Format**: A list of task description strings, e.g., ["Analyze user needs", "Design system architecture", "Implement core module"]
- **Automatic Status Assignment**:
  - First task → `"in_progress"`
  - All subsequent tasks → `"pending"`
- **Plan Replacement Rule**: Provide **only the new tasks that should be executed next**. Do not include completed, obsolete, or irrelevant tasks.
- **Plan Quality Requirements**:
  - Tasks must be specific, actionable, and verifiable
  - Break work into logical phases (chronological or dependency-based)
  - Define clear milestones and deliverable standards
  - Avoid vague, ambiguous, or non-executable descriptions
- **Do NOT Use When**:
  - The task is simple (<3 steps)
  - The request is conversational, informational, or a one-off query
  - You only need to mark a task as complete (use `finish_sub_plan` instead)

## 2. finish_sub_plan: Mark Current Task as Complete
- **Primary Purpose**: Confirm the current `"in_progress"` task is fully done, mark it as `"done"`, and automatically promote the first `"pending"` task to `"in_progress"`.
- **Call Only If ALL Conditions Are Met**:
  - The subtask has been **fully executed**
  - All specified requirements have been satisfied
  - There are no unresolved errors, omissions, or blockers
  - The output meets quality standards and has been verified
- **Automatic Behavior**:
  - No parameters needed—status transitions are handled internally
  - If no `"pending"` tasks remain, the plan ends naturally
- **Never Call If**:
  - The task is partially complete
  - Known issues or defects remain
  - Execution was blocked due to missing resources or dependencies
  - The result fails to meet expected quality

{read_plan_system_prompt}

## Task Status Rules (Only These Three Are Valid)
- **`"pending"`**: Task not yet started
- **`"in_progress"`**: Currently being executed (exactly one allowed at any time)
- **`"done"`**: Fully completed and verified  
> ⚠️ No other status values (e.g., "completed", "failed", "blocked") are permitted.

## General Usage Principles
1. **Execute simple tasks directly**: If a request can be fulfilled in 1–2 steps, do not create a plan—just complete it.
2. **Decompose thoughtfully**: Break complex work into clear, independent, trackable subtasks.
3. **Manage status rigorously**:
   - Always maintain exactly one `"in_progress"` task while work is ongoing
   - Call `finish_sub_plan` immediately after task completion—never delay
4. **Plan modification = full replacement**: Never edit individual tasks. To adjust the plan, use `write_plan` with a new list of remaining tasks.
5. **Respect user intent**: If the user explicitly asks for a plan—even for a simpler task—honor the request and create one.
"""


class PlanMiddleware(AgentMiddleware):
    """Middleware that provides plan management capabilities to agents.

    This middleware adds a `write_plan` and `finish_sub_plan` (and `read_plan` optional) tool that allows agents to create and manage
    structured plan lists for complex multi-step operations. It's designed to help
    agents track progress, organize complex tasks, and provide users with visibility
    into task completion status.

    The middleware automatically injects system prompts that guide the agent on how to use the plan functionality effectively.

    Example:
        ```python
        from langchain_dev_utils.agents.middleware.plan import PlanMiddleware
        from langchain_dev_utils.agents import create_agent

        agent = create_agent("vllm:qwen3-4b", middleware=[PlanMiddleware()])

        # Agent now has access to write_plan tool and plan state tracking
        result = await agent.invoke({"messages": [HumanMessage("Help me refactor my codebase")]})

        print(result["plan"])  # Array of plan items with status tracking
        ```

    Args:
        system_prompt: Custom system prompt to guide the agent on using the plan tool.
            If not provided, uses the default `_PLAN_MIDDLEWARE_SYSTEM_PROMPT`.
        tools: List of tools to be added to the agent. The tools must be created by `create_write_plan_tool`, `create_finish_sub_plan_tool`, and `create_read_plan_tool`(optional).
    """

    state_schema = PlanState

    def __init__(
        self,
        *,
        system_prompt: str = _PLAN_MIDDLEWARE_SYSTEM_PROMPT,
        tools: Optional[list[BaseTool]] = None,
    ) -> None:
        """Initialize the TodoListMiddleware with optional custom prompts.

        Args:
            system_prompt: Custom system prompt to guide the agent on using the todo tool.
            tools: Custom tools to be added to the agent. The tools must be created by `create_write_plan_tool`, `create_finish_sub_plan_tool`, and `create_read_plan_tool`(optional).
        """
        super().__init__()
        self.system_prompt = system_prompt

        if tools is None:
            tools = [
                create_write_plan_tool(),
                create_finish_sub_plan_tool(),
                create_read_plan_tool(),
            ]
        self.tools = tools

        num = 2
        read_plan_system = ""
        if len(self.tools) == 3:
            num = 3
            read_plan_system = _READ_PLAN_SYSTEM_PROMPT

        self.system_prompt = self.system_prompt.format(
            num=num, read_plan_system_prompt=read_plan_system
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Update the system prompt to include the todo system prompt."""
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
        """Update the system prompt to include the todo system prompt (async version)."""
        request.system_prompt = (
            request.system_prompt + "\n\n" + self.system_prompt
            if request.system_prompt
            else self.system_prompt
        )
        return await handler(request)
