from dotenv import load_dotenv
from langchain.tools import tool
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage, ToolMessage

from langchain_dev_utils.agents import create_agent
from langchain_dev_utils.agents.middleware import ModelRouterMiddleware
from langchain_dev_utils.chat_models import batch_register_model_provider

load_dotenv()

batch_register_model_provider(
    [
        {
            "provider": "dashscope",
            "chat_model": ChatTongyi,
        },
        {"provider": "zai", "chat_model": "openai-compatible"},
    ]
)


@tool
def run_python_code(code: str) -> str:
    """Run python code"""
    return "Program run completed without errors"


def test_model_router_middleware():
    agent = create_agent(
        model="dashscope:qwen3-max",
        tools=[run_python_code],
        middleware=[
            ModelRouterMiddleware(
                router_model="dashscope:qwen-flash",
                model_list=[
                    {
                        "model_name": "dashscope:qwen3-max",
                        "model_description": "The most intelligent large model",
                    },
                    {
                        "model_name": "zai:glm-4.5",
                        "model_description": "The model with the strongest coding performance",
                        "tools": [run_python_code],
                    },
                ],
            )
        ],
    )
    response = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Implement a simple hello world program without thinking, and finally use the **run_python_code** tool to run the code"
                )
            ]
        }
    )
    assert response
    assert response["messages"][-1].response_metadata.get("model_name") == "glm-4.5"
    assert isinstance(response["messages"][-2], ToolMessage)


async def test_model_router_middleware_async():
    agent = create_agent(
        model="dashscope:qwen3-max",
        tools=[run_python_code],
        middleware=[
            ModelRouterMiddleware(
                router_model="dashscope:qwen-flash",
                model_list=[
                    {
                        "model_name": "dashscope:qwen3-max",
                        "model_description": "The most intelligent large model",
                    },
                    {
                        "model_name": "zai:glm-4.5",
                        "model_description": "The model with the strongest coding performance",
                        "tools": [run_python_code],
                    },
                ],
            )
        ],
    )
    response = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="Implement a simple hello world program without thinking, and finally use the **run_python_code** tool to run the code"
                )
            ]
        }
    )
    assert response
    assert response["messages"][-1].response_metadata.get("model_name") == "glm-4.5"
    assert isinstance(response["messages"][-2], ToolMessage)
