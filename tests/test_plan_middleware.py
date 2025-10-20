from langchain_core.messages import AIMessage, HumanMessage

from langchain_dev_utils.agents import create_agent
from langchain_dev_utils.agents.middleware.plan import (
    PlanMiddleware,
    create_update_plan_tool,
    create_write_plan_tool,
)
from langchain_dev_utils.chat_models import register_model_provider
from langchain_dev_utils.tool_calling.utils import has_tool_calling, parse_tool_calling

register_model_provider(
    provider_name="zai",
    chat_model="openai-compatible",
    base_url="https://open.bigmodel.cn/api/paas/v4",
)


def test_plan_tool():
    write_plan_tool = create_write_plan_tool()
    update_plan_tool = create_update_plan_tool()

    assert write_plan_tool.name == "write_plan"
    assert update_plan_tool.name == "update_plan"

    write_file_tool_with_name = create_write_plan_tool(name="write_plan_list")
    update_file_tool_with_name = create_update_plan_tool(name="update_plan_list")
    assert write_file_tool_with_name.name == "write_plan_list"
    assert update_file_tool_with_name.name == "update_plan_list"

    write_file_tool_with_description = create_write_plan_tool(
        description="the tool for writing plan list"
    )
    update_file_tool_with_description = create_update_plan_tool(
        description="the tool for updating plan list"
    )
    assert (
        write_file_tool_with_description.description == "the tool for writing plan list"
    )
    assert (
        update_file_tool_with_description.description
        == "the tool for updating plan list"
    )

    write_file_tool_with_name_and_description = create_write_plan_tool(
        name="write_plan_list_tool", description="the tool for writing plan list"
    )
    update_file_tool_with_name_and_description = create_update_plan_tool(
        name="update_plan_list_tool", description="the tool for updating plan list"
    )
    assert write_file_tool_with_name_and_description.name == "write_plan_list_tool"
    assert update_file_tool_with_name_and_description.name == "update_plan_list_tool"
    assert (
        write_file_tool_with_name_and_description.description
        == "the tool for writing plan list"
    )
    assert (
        update_file_tool_with_name_and_description.description
        == "the tool for updating plan list"
    )


def test_plan_middleware():
    plan_middleware = PlanMiddleware()

    agent = create_agent(
        model="zai:glm-4.5",
        middleware=[plan_middleware],
        system_prompt="请使用write_plan和update_plan工具制定一个计划，计划数量必须是3个，然后依次执行这些计划用update_plan工具更新计划。最终确保所有计划的状态都为done",
    )

    result = agent.invoke({"messages": [HumanMessage(content="请开始执行吧")]})

    assert result["plan"]
    assert len(result["plan"]) == 3
    assert all([plan["status"] == "done" for plan in result["plan"]])

    write_plan_count = 0
    update_plan_count = 0
    for message in result["messages"]:
        if isinstance(message, AIMessage) and has_tool_calling(message):
            name, _ = parse_tool_calling(message, first_tool_call_only=True)
            if name == "write_plan":
                write_plan_count += 1
            elif name == "update_plan":
                update_plan_count += 1

    assert write_plan_count == 1
    assert update_plan_count == 3
