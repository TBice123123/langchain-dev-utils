from typing import Annotated, Literal, Optional

from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState
from langchain_core.tools import InjectedToolCallId, BaseTool, tool
from langgraph.types import Command
from typing_extensions import TypedDict

_DEFAULT_WRITE_PLAN_TOOL_DESCRIPTION = """
A tool for writing initial plan — can only be used once, at the very beginning. 
Use update_plan for subsequent modifications.

Args:
    plan: The list of plan items to write. Each string in the list represents 
          the content of one plan item.
"""

_DEFAULT_UPDATE_PLAN_TOOL_DESCRIPTION = """
A tool for updating the status of plan tasks. Can be called multiple times to track task progress.

Args:
    update_plans: A list of plan items to update. Each item is a dictionary containing 
                  the following fields:
                  - content: str — The exact content of the plan task. Must match an 
                    existing task verbatim.
                  - status: str — The task status. Must be either "in_progress" or "done".

Usage Guidelines:
- Only pass the tasks whose status needs to be updated — no need to include all tasks.
- Each call must include at least one task with status "done" AND at least one task with 
  status "in_progress":
  - Mark completed tasks as "done"
  - Mark the next tasks to work on as "in_progress"
- The "content" field must exactly match the content of an existing task 
  (case-sensitive, whitespace-sensitive).

Example:
Suppose the current task list is:
- Task 1 (in_progress)
- Task 2 (pending)
- Task 3 (pending)

When "Task 1" is completed and you are ready to start "Task 2", pass in:
[
    {"content": "Task 1", "status": "done"},
    {"content": "Task 2", "status": "in_progress"}
]
"""


class Plan(TypedDict):
    content: str
    status: Literal["pending", "in_progress", "done"]


class PlanStateMixin(TypedDict):
    plan: list[Plan]


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
    def write_plan(plan: list[str], tool_call_id: Annotated[str, InjectedToolCallId]):
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
                        tool_call_id=tool_call_id,
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
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[PlanStateMixin, InjectedState],
    ):
        plan_list = state.get("plan", [])

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
                        content="Plan updated successfully", tool_call_id=tool_call_id
                    )
                ],
            }
        )

    return update_plan
