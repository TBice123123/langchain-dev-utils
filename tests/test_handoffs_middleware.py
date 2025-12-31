from datetime import datetime

from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from langchain_dev_utils.agents import create_agent
from langchain_dev_utils.agents.middleware import (
    HandoffAgentMiddleware,
)
from langchain_dev_utils.agents.middleware.handoffs import AgentConfig

from langchain_dev_utils.chat_models import load_chat_model


@tool
def get_current_time() -> str:
    """Get the current time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def run_code(code: str) -> str:
    """Run the given code."""
    return "Running code successfully"


agents_config: dict[str, AgentConfig] = {
    "time_agent": {
        "model": "zai:glm-4.5",
        "prompt": (
            "You are a time assistant. You can answer users' time-related questions. "
            "If the current question is not time-related, please transfer to another assistant."
        ),
        "tools": [get_current_time],
        "handoffs": ["talk_agent"],
    },
    "talk_agent": {
        "prompt": (
            "You are a conversational assistant. You can answer users' questions. "
            "If the current question is a time query, please transfer to the time assistant; "
            "if it's a code-related question, please transfer to the code assistant."
        ),
        "default": True,
        "handoffs": "all",
    },
    "code_agent": {
        "model": load_chat_model("dashscope:qwen3-coder-plus"),
        "prompt": (
            "You are a code assistant. You can answer users' code-related questions. "
            "If the current question is not code-related, please transfer to another assistant."
        ),
        "tools": [run_code],
        "handoffs": ["talk_agent"],
    },
}
custom_tool_descriptions: dict[str, str] = {
    "time_agent": "transfer to the time agent to answer time-related questions",
    "talk_agent": "transfer to the talk agent to answer user questions",
    "code_agent": "transfer to the code agent to answer code-related questions",
}


def test_handoffs_middleware():
    agent = create_agent(
        model="dashscope:qwen3-max",
        middleware=[HandoffAgentMiddleware(agents_config, custom_tool_descriptions)],
        tools=[
            get_current_time,
            run_code,
        ],
        checkpointer=InMemorySaver(),
    )

    response = agent.invoke(
        {"messages": [HumanMessage(content="get current time")]},
        config={"configurable": {"thread_id": "123"}},
    )

    assert response
    assert response["messages"][-1].response_metadata.get("model_name") == "glm-4.5"
    assert isinstance(response["messages"][-2], ToolMessage)
    assert "active_agent" in response and response["active_agent"] == "time_agent"

    response = agent.invoke(
        {
            "messages": [
                HumanMessage(content="Implement a simple bubble sort in Python")
            ]
        },
        config={"configurable": {"thread_id": "123"}},
    )
    assert response
    assert (
        response["messages"][-1].response_metadata.get("model_name")
        == "qwen3-coder-plus"
    )
    assert isinstance(response["messages"][-2], ToolMessage)
    assert "active_agent" in response and response["active_agent"] == "code_agent"


async def test_handoffs_middleware_async():
    agent = create_agent(
        model="dashscope:qwen3-max",
        middleware=[HandoffAgentMiddleware(agents_config, custom_tool_descriptions)],
        tools=[
            get_current_time,
            run_code,
        ],
        checkpointer=InMemorySaver(),
    )

    response = await agent.ainvoke(
        {"messages": [HumanMessage(content="get current time")]},
        config={"configurable": {"thread_id": "234"}},
    )

    assert response
    assert response["messages"][-1].response_metadata.get("model_name") == "glm-4.5"
    assert isinstance(response["messages"][-2], ToolMessage)
    assert "active_agent" in response and response["active_agent"] == "time_agent"

    response = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(content="Implement a simple bubble sort in Python")
            ]
        },
        config={"configurable": {"thread_id": "234"}},
    )
    assert response
    assert (
        response["messages"][-1].response_metadata.get("model_name")
        == "qwen3-coder-plus"
    )
    assert isinstance(response["messages"][-2], ToolMessage)
    assert "active_agent" in response and response["active_agent"] == "code_agent"
