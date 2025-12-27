from datetime import datetime
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_dev_utils.agents import create_agent
from langchain_dev_utils.agents.middleware import (
    HandoffsAgentMiddleware,
    create_handoffs_tool,
)
from langchain_dev_utils.agents.middleware.handoffs import AgentConfig
from tests.utils.register import register_all_model_providers

register_all_model_providers()

transfer_time_agent = create_handoffs_tool("time_agent")
transfer_talk_agent = create_handoffs_tool("talk_agent")


@tool
def get_current_time() -> str:
    """Get the current time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


agents_config: dict[str, AgentConfig] = {
    "time_agent": {
        "model": "zai:glm-4.5",
        "prompt": "你是一个时间助手，你可以回答用户的时间相关问题",
        "tools": [transfer_talk_agent, get_current_time],
    },
    "talk_agent": {
        "prompt": "你是一个对话助手，你可以回答用户的问题",
        "tools": [transfer_time_agent],
        "default": True,
    },
}


def test_handoffs_middleware():
    agent = create_agent(
        model="dashscope:qwen-flash",
        middleware=[HandoffsAgentMiddleware(agents_config)],
        tools=[
            get_current_time,
            transfer_time_agent,
            transfer_talk_agent,
        ],
    )

    response = agent.invoke({"messages": [HumanMessage(content="get current time")]})

    assert response
    assert response["messages"][-1].response_metadata.get("model_name") == "glm-4.5"
    assert isinstance(response["messages"][-2], ToolMessage)
    assert "active_agent" in response and response["active_agent"] == "time_agent"


async def test_handoffs_middleware_async():
    agent = create_agent(
        model="dashscope:qwen-flash",
        middleware=[HandoffsAgentMiddleware(agents_config)],
        tools=[
            get_current_time,
            transfer_time_agent,
            transfer_talk_agent,
        ],
    )

    response = await agent.ainvoke(
        {"messages": [HumanMessage(content="get current time")]}
    )

    assert response
    assert response["messages"][-1].response_metadata.get("model_name") == "glm-4.5"
    assert isinstance(response["messages"][-2], ToolMessage)
    assert "active_agent" in response and response["active_agent"] == "time_agent"
