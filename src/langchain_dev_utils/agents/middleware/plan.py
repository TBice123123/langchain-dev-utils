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
A tool for writing initial plan — can only be used once, at the very beginning. 
Use update_plan for subsequent modifications.

Args:
    plan: The list of plan items to write. Each string in the list represents 
          the content of one plan item.
"""

_DEFAULT_UPDATE_PLAN_TOOL_DESCRIPTION = (
    WRITE_TODOS_TOOL_DESCRIPTION
) = """Use this tool to create and manage a structured task list for your current work session. This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user.

Only use this tool if you think it will be helpful in staying organized. If the user's request is trivial and takes less than 3 steps, it is better to NOT use this tool and just do the task directly.

## Tool Division
- **write_plan**: Used to create an initial plan framework for complex tasks
- **update_plan**: Used to dynamically update progress and adjust the plan during task execution

## Usage Scenarios
Use this tool in the following situations:

1. Complex multi-step tasks - When a task requires 3 or more distinct steps or actions
2. Project-level complex tasks - Work involving multiple phases, dependencies, or resource coordination
3. User explicitly requests a task list - When the user directly asks you to use the todo list
4. User provides multiple tasks - When users provide a list of things to be done (numbered or comma-separated)
5. The plan may need dynamic adjustments based on execution results

## Plan Creation (write_plan)
**Usage Scenarios**:
- Need to create a complete execution framework at project initiation
- Complex tasks require phased approaches and clear dependencies
- Long-term work requires resource planning and milestone setting

**Important Specifications**:
- When creating the initial plan, the FIRST task should automatically be set to "in_progress" status
- No additional status updates are needed during plan creation - the initial state is properly configured

**Plan Elements**:
- Phase division: Divide work phases according to logical or chronological order
- Dependencies: Clarify sequential dependencies and relationships between tasks
- Milestones: Set key nodes and deliverable standards
- Resource planning: Estimate required time, tools, or external resources

## Plan Update (update_plan)
**Usage Scenarios**:
- Task status changes need real-time reflection
- Need to adjust execution path when encountering obstacles
- User proposes new requirements or modifies existing ones
- Discover more optimized execution methods

**Critical Update Principle**:
- **ONLY pass the specific tasks that require modification** - do not resend the entire plan
- The system will intelligently merge your partial updates with the existing plan
- This ensures efficient updates and prevents unnecessary data transfer

**Update Principles**:
- Real-time updates: Update the plan immediately after task status changes
- Partial updates: Only modify the tasks that actually changed status or content
- Transparent progress: Accurately reflect completed, in-progress, and pending tasks
- Flexible adjustments: Reasonably adjust subsequent arrangements based on actual situation

## Task States and Management Standards

1. **Available Task States**:
   - **"pending"**: Task not yet started (waiting to be processed)
   - **"in_progress"**: Currently actively working on this task
   - **"done"**: Task fully completed and verified
   - *Note: These three states are the ONLY available options - no other status values are accepted*

2. **Task Management**:
   - First task is automatically set to "in_progress" when plan is created via write_plan
   - Mark tasks as "done" immediately upon completion using update_plan
   - Update only the specific tasks that change status - no need to resend unchanged tasks
   - Always maintain at least one "in_progress" task while work is ongoing

3. **Completion Standards**:
   - ONLY mark a task as "done" when you have FULLY accomplished it
   - Maintain "in_progress" status when encountering obstacles or partial completion
   - Never mark a task as "done" if:
     * There are unresolved errors or issues
     * Work is partial or incomplete
     * Quality standards haven't been met

4. **Task Breakdown Principles**:
   - Create specific, actionable items
   - Break complex tasks into smaller, manageable steps
   - Use clear, descriptive task names

## Prohibited Scenarios
Avoid using this tool in the following cases:
1. When there is only a single simple task
2. When the task can be completed in less than 3 simple steps
3. For purely conversational or informational tasks
4. For temporary simple queries or operations

**Important Reminder**: 
- Use write_plan for initial plan creation - first task automatically becomes "in_progress"
- Use update_plan for subsequent changes - only pass the specific tasks that need modification
- Remember: Only three valid states exist - "pending", "in_progress", and "done"
- For simple tasks with clear execution paths, execute directly without planning
"""


class Plan(TypedDict):
    content: str
    status: Literal["pending", "in_progress", "done"]


class PlanState(AgentState):
    plan: NotRequired[list[Plan]]


def create_write_plan_tool(
    name: Optional[str] = None,
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
        >>> from langchain_dev_utils.agents.plan import create_write_plan_tool
        >>> write_plan_tool = create_write_plan_tool()
    """

    @tool(
        name_or_callable=name or "write_plan",
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


def create_update_plan_tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    message_key: Optional[str] = None,
) -> BaseTool:
    """Create a tool for updating plan tasks.

    This function creates a tool that allows agents to update the status of tasks
    in a plan. Tasks can be marked as "in_progress" or "done" to track progress.

    Args:
        name: The name of the tool. Defaults to "update_plan".
        description: The description of the tool. Uses default description if not provided.
        message_key: The key of the message to be updated. Defaults to "messages".

    Returns:
        BaseTool: The tool for updating plan tasks.

    Example:
        Basic usage:
        >>> from langchain_dev_utils.agents.plan import create_update_plan_tool
        >>> update_plan_tool = create_update_plan_tool()
    """

    @tool(
        name_or_callable=name or "update_plan",
        description=description or _DEFAULT_UPDATE_PLAN_TOOL_DESCRIPTION,
    )
    def update_plan(
        update_plans: list[Plan],
        runtime: ToolRuntime,
    ):
        plan_list = runtime.state.get("plan", [])

        updated_plan_list = []

        for update_plan in update_plans:
            for plan in plan_list:
                if plan["content"] == update_plan["content"]:
                    plan["status"] = update_plan["status"]
                    updated_plan_list.append(plan)

        if len(updated_plan_list) < len(update_plans):
            raise ValueError(
                "Not fullly updated plan, missing:"
                + ",".join(
                    [
                        plan["content"]
                        for plan in update_plans
                        if plan not in updated_plan_list
                    ]
                )
                + "\nPlease check the plan list, the current plan list is:"
                + "\n".join(
                    [plan["content"] for plan in plan_list if plan["status"] != "done"]
                )
            )
        msg_key = message_key or "messages"

        return Command(
            update={
                "plan": plan_list,
                msg_key: [
                    ToolMessage(
                        content="Plan updated successfully",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    return update_plan


WRITE_PLAN_TOOL_DESCRIPTION = """Use this tool to create and manage a structured task list for your current work session. This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user.

Only use this tool if you think it will be helpful in staying organized. If the user's request is trivial and takes less than 3 steps, it is better to NOT use this tool and just do the task directly.

## Tool Division
- **write_plan**: Used to create an initial plan framework for complex tasks
- **update_plan**: Used to dynamically update progress and adjust the plan during task execution

## Usage Scenarios
Use this tool in the following situations:

1. Complex multi-step tasks - When a task requires 3 or more distinct steps or actions
2. Project-level complex tasks - Work involving multiple phases, dependencies, or resource coordination
3. User explicitly requests a task list - When the user directly asks you to use the todo list
4. User provides multiple tasks - When users provide a list of things to be done (numbered or comma-separated)
5. The plan may need dynamic adjustments based on execution results

## Plan Creation (write_plan)
**Usage Scenarios**:
- Need to create a complete execution framework at project initiation
- Complex tasks require phased approaches and clear dependencies
- Long-term work requires resource planning and milestone setting

**Plan Elements**:
- Phase division: Divide work phases according to logical or chronological order
- Dependencies: Clarify sequential dependencies and relationships between tasks
- Milestones: Set key nodes and deliverable standards
- Resource planning: Estimate required time, tools, or external resources

## Plan Update (update_plan)
**Usage Scenarios**:
- Task status changes need real-time reflection
- Need to adjust execution path when encountering obstacles
- User proposes new requirements or modifies existing ones
- Discover more optimized execution methods

**Update Principles**:
- Real-time updates: Update the plan immediately after task status changes
- Transparent progress: Accurately reflect completed, in-progress, and pending tasks
- Flexible adjustments: Reasonably adjust subsequent arrangements based on actual situation
- Maintain consistency: Ensure overall logical consistency after plan adjustments

## Task States and Management Standards

1. **Task States**:
   - pending: Task not yet started
   - in_progress: Currently working on (multiple parallel tasks allowed in this state)
   - completed: Task finished successfully

2. **Task Management**:
   - Set the first task to "in_progress" immediately after creating the plan
   - Mark tasks as complete immediately and update the next task status
   - Completely remove tasks that are no longer relevant
   - Always maintain at least one "in_progress" task to show work status

3. **Completion Standards**:
   - ONLY mark a task as completed when you have FULLY accomplished it
   - Maintain "in_progress" status and describe problems when encountering obstacles
   - Never mark a task as completed if:
     * There are unresolved errors or issues
     * Work is partial or incomplete
     * Quality standards haven't been met

4. **Task Breakdown Principles**:
   - Create specific, actionable items
   - Break complex tasks into smaller, manageable steps
   - Use clear, descriptive task names

## Prohibited Scenarios
Avoid using this tool in the following cases:
1. When there is only a single simple task
2. When the task can be completed in less than 3 simple steps
3. For purely conversational or informational tasks
4. For temporary simple queries or operations

**Important Reminder**: If the task is clearly simple and the execution path is straightforward, execute it directly without creating a plan. Planning tools should be used for truly complex work scenarios that require structured management.
"""


class PlanMiddleware(AgentMiddleware):
    """Middleware that provides plan management capabilities to agents.

    This middleware adds a `write_plan` and `update_plan` tool that allows agents to create and manage
    structured plan lists for complex multi-step operations. It's designed to help
    agents track progress, organize complex tasks, and provide users with visibility
    into task completion status.

    The middleware automatically injects system prompts that guide the agent on when
    and how to use the plan functionality effectively.

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
            If not provided, uses the default `WRITE_PLAN_SYSTEM_PROMPT`.
        tools: List of tools to be added to the agent. The tools must be created by `create_write_plan_tool` and `create_update_plan_tool`.
    """

    state_schema = PlanState

    def __init__(
        self,
        *,
        system_prompt: str = WRITE_PLAN_TOOL_DESCRIPTION,
        tools: Optional[list[BaseTool]] = None,
    ) -> None:
        """Initialize the TodoListMiddleware with optional custom prompts.

        Args:
            system_prompt: Custom system prompt to guide the agent on using the todo tool.
            tool_description: Custom description for the write_todos tool.
        """
        super().__init__()
        self.system_prompt = system_prompt

        if tools is None:
            tools = [create_write_plan_tool(), create_update_plan_tool()]
        self.tools = tools

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
