from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from langchain_dev_utils.agents import create_agent
from langchain_dev_utils.agents.middleware.plan import (
    PlanMiddleware,
    create_finish_sub_plan_tool,
    create_read_plan_tool,
    create_write_plan_tool,
)
from langchain_dev_utils.chat_models import register_model_provider
from langchain_dev_utils.tool_calling.utils import has_tool_calling, parse_tool_calling

load_dotenv()
register_model_provider(provider_name="zai", chat_model="openai-compatible")


def test_plan_tool():
    write_plan_tool = create_write_plan_tool()
    finish_sub_plan_tool = create_finish_sub_plan_tool()
    read_plan_tool = create_read_plan_tool()

    assert write_plan_tool.name == "write_plan"
    assert finish_sub_plan_tool.name == "finish_sub_plan"
    assert read_plan_tool.name == "read_plan"

    write_plan_tool = create_write_plan_tool(
        description="the tool for writing plan list"
    )
    finish_sub_plan_tool = create_finish_sub_plan_tool(
        description="the tool for finish sub plan"
    )
    read_plan_tool = create_read_plan_tool(description="the tool for reading plan list")

    assert write_plan_tool.description == "the tool for writing plan list"
    assert finish_sub_plan_tool.description == "the tool for finish sub plan"
    assert read_plan_tool.description == "the tool for reading plan list"


def test_plan_middleware():
    plan_middleware = PlanMiddleware()

    agent = create_agent(model="zai:glm-4.5", middleware=[plan_middleware])

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Please use the `write_plan` tool to specify a plan, and the number of plans must be 3. Then execute these plans one by one using the `finish_sub_plan` tool to update the plan status. Finally, make sure that the status of all plans is `done`"
                )
            ]
        }
    )

    assert result["plan"]
    assert len(result["plan"]) == 3
    assert all([plan["status"] == "done" for plan in result["plan"]])

    write_plan_count = 0
    finish_sub_plan_count = 0
    for message in result["messages"]:
        if isinstance(message, AIMessage) and has_tool_calling(message):
            name, _ = parse_tool_calling(message, first_tool_call_only=True)
            if name == "write_plan":
                write_plan_count += 1
            elif name == "finish_sub_plan":
                finish_sub_plan_count += 1

    assert write_plan_count == 1
    assert finish_sub_plan_count == 3
